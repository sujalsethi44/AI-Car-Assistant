import pandas as pd
import faiss
import numpy as np
import openai
import re
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE = os.getenv('API_BASE')
API_KEY = os.getenv('API_KEY')
API_VERSION = os.getenv('API_VERSION')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')

# Configure OpenAI for Azure
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION

CSV_PATH = "car_descriptions_embeddings.csv"

@dataclass
class CarInfo:
    """Structured car information extracted from description"""
    description: str
    brand: str
    model: str
    year: int
    price: float
    fuel_type: str
    transmission: str
    mileage: int
    ownership: str
    condition_score: float
    estimated_msrp: float
    discount_percentage: float

def parse_car_description(description: str) -> CarInfo:
    """Extract structured information from car description"""
    # Extract year
    year_match = re.search(r'(\d{4}) model', description)
    year = int(year_match.group(1)) if year_match else 2010
    
    # Extract brand and model
    brand_model_match = re.search(r'model ([A-Za-z]+)\s+([A-Za-z0-9\s\.\-]+?)\s+is', description)
    if brand_model_match:
        brand = brand_model_match.group(1)
        model = brand_model_match.group(2).strip()
    else:
        brand = "Unknown"
        model = "Unknown"
    
    # Extract price
    price_match = re.search(r'₹([\d\.]+)\s*(lakh|thousand)?', description)
    if price_match:
        price_val = float(price_match.group(1))
        unit = price_match.group(2)
        if unit == 'lakh':
            price = price_val * 100000
        elif unit == 'thousand':
            price = price_val * 1000
        else:
            price = price_val
    else:
        price = 0
    
    # Extract fuel type
    fuel_type = "petrol" if "petrol-powered" in description else "diesel" if "diesel-powered" in description else "unknown"
    
    # Extract transmission
    transmission = "manual" if "manual transmission" in description else "automatic" if "automatic transmission" in description else "unknown"
    
    # Extract mileage
    mileage_match = re.search(r'driven for ([\d,]+)\s*km', description)
    mileage = int(mileage_match.group(1).replace(',', '')) if mileage_match else 0
    
    # Extract ownership
    ownership_match = re.search(r'(first|second|third) owner', description)
    ownership = ownership_match.group(1) if ownership_match else "unknown"
    
    # Calculate condition score (0-10)
    condition_score = calculate_condition_score(year, mileage, ownership)
    
    # Estimate MSRP and discount
    estimated_msrp = estimate_msrp(brand, model, year)
    discount_percentage = ((estimated_msrp - price) / estimated_msrp * 100) if estimated_msrp > 0 else 0
    
    return CarInfo(
        description=description,
        brand=brand,
        model=model,
        year=year,
        price=price,
        fuel_type=fuel_type,
        transmission=transmission,
        mileage=mileage,
        ownership=ownership,
        condition_score=condition_score,
        estimated_msrp=estimated_msrp,
        discount_percentage=discount_percentage
    )

def calculate_condition_score(year: int, mileage: int, ownership: str) -> float:
    """Calculate condition score from 0-10 based on year, mileage, and ownership"""
    current_year = 2024
    age = current_year - year
    
    # Base score starts at 10
    score = 10.0
    
    # Deduct for age (0.5 points per year)
    score -= age * 0.5
    
    # Deduct for mileage (1 point per 20,000 km)
    score -= (mileage / 20000) * 1.0
    
    # Deduct for ownership
    if ownership == "second":
        score -= 1.0
    elif ownership == "third":
        score -= 2.0
    
    return max(0, min(10, score))

def estimate_msrp(brand: str, model: str, year: int) -> float:
    """Estimate original MSRP based on brand, model, and year"""
    # Simplified MSRP estimation
    base_prices = {
        "maruti": {
            "800": 250000, "alto": 300000, "wagon r": 400000,
            "swift": 500000, "baleno": 600000
        },
        "hyundai": {
            "verna": 800000, "xcent": 600000, "creta": 1000000,
            "i10": 400000, "i20": 600000
        },
        "honda": {
            "amaze": 600000, "city": 900000, "jazz": 700000
        },
        "tata": {
            "indigo": 500000, "nano": 200000, "bolt": 600000
        },
        "datsun": {
            "redigo": 350000, "go": 400000
        }
    }
    
    brand_lower = brand.lower()
    model_lower = model.lower()
    
    # Find matching model
    base_price = 500000  # default
    if brand_lower in base_prices:
        for model_key, price in base_prices[brand_lower].items():
            if model_key in model_lower:
                base_price = price
                break
    
    # Adjust for year (inflation for base prices)
    current_year = 2024
    age = current_year - year
    
    return base_price * (1 + age * 0.05)  # Inflation adjustment

def load_data(csv_path: str) -> Tuple[pd.DataFrame, List[CarInfo]]:
    """Load CSV and parse car information"""
    df = pd.read_csv(csv_path)
    # Ensure embedding is a list of floats
    df["embedding"] = df["embedding"].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])
    
    # Parse car information
    car_infos = []
    for _, row in df.iterrows():
        car_info = parse_car_description(row['car_description'])
        car_infos.append(car_info)
    
    return df, car_infos

class CarSearchEngine:
    def __init__(self, csv_path: str):
        self.df, self.car_infos = load_data(csv_path)
        vector_size = len(self.df["embedding"].iloc[0])
        self.index = faiss.IndexFlatL2(vector_size)
        embeddings = self.df["embedding"].tolist()
        self.index.add(np.array(embeddings, dtype="float32"))
        
    def filter_by_criteria(self, car_infos: List[CarInfo], 
                          min_price: Optional[float] = None,
                          max_price: Optional[float] = None,
                          brands: Optional[List[str]] = None,
                          fuel_types: Optional[List[str]] = None,
                          min_condition: Optional[float] = None,
                          max_mileage: Optional[int] = None) -> List[CarInfo]:
        """Filter cars by various criteria"""
        filtered = car_infos.copy()
        
        if min_price is not None:
            filtered = [car for car in filtered if car.price >= min_price]
        if max_price is not None:
            filtered = [car for car in filtered if car.price <= max_price]
        if brands:
            filtered = [car for car in filtered if car.brand.lower() in [b.lower() for b in brands]]
        if fuel_types:
            filtered = [car for car in filtered if car.fuel_type.lower() in [f.lower() for f in fuel_types]]
        if min_condition is not None:
            filtered = [car for car in filtered if car.condition_score >= min_condition]
        if max_mileage is not None:
            filtered = [car for car in filtered if car.mileage <= max_mileage]
            
        return filtered
    
    def hybrid_search(self, query_vector: List[float], 
                     min_price: Optional[float] = None,
                     max_price: Optional[float] = None,
                     brands: Optional[List[str]] = None,
                     fuel_types: Optional[List[str]] = None,
                     min_condition: Optional[float] = None,
                     max_mileage: Optional[int] = None,
                     top_k: int = 5) -> List[CarInfo]:
        """Enhanced hybrid search with filtering"""
        # Vector search
        query = np.array([query_vector], dtype="float32")
        D, I = self.index.search(query, len(self.car_infos))  # Get all results
        
        # Get corresponding car infos
        search_results = [(self.car_infos[i], D[0][idx]) for idx, i in enumerate(I[0])]
        
        # Apply filters
        filtered_cars = [car for car, _ in search_results]
        filtered_cars = self.filter_by_criteria(
            filtered_cars, min_price, max_price, brands, fuel_types, min_condition, max_mileage
        )
        
        # Re-rank filtered results by similarity score
        final_results = []
        for car, score in search_results:
            if car in filtered_cars:
                final_results.append((car, score))
                if len(final_results) >= top_k:
                    break
        
        return [car for car, _ in final_results]
    
    def get_trade_in_estimate(self, car_info: CarInfo) -> float:
        """Estimate trade-in value (typically 70-80% of current market value)"""
        market_value = car_info.price
        trade_in_percentage = 0.75 - (car_info.condition_score / 100)  # Better condition = higher trade-in
        return market_value * max(0.6, trade_in_percentage)
    
    def format_car_result(self, car: CarInfo, include_trade_in: bool = False) -> str:
        """Format car information for display"""
        condition_text = "Excellent" if car.condition_score >= 8 else "Good" if car.condition_score >= 6 else "Fair" if car.condition_score >= 4 else "Poor"
        
        result = f"""{car.brand} {car.model} ({car.year})
Price: ₹{car.price:,.0f}
Fuel: {car.fuel_type.title()}
Transmission: {car.transmission.title()}
Mileage: {car.mileage:,} km
Ownership: {car.ownership.title()} owner
Condition: {condition_text} ({car.condition_score:.1f}/10)
MSRP Discount: {car.discount_percentage:.1f}% off estimated original price"""
        
        if include_trade_in:
            trade_in = self.get_trade_in_estimate(car)
            result += f"\nEstimated Trade-in Value: ₹{trade_in:,.0f}"
        
        return result

def extract_search_criteria(user_input: str) -> Dict:
    """Extract search criteria from user input using simple keyword matching"""
    criteria = {}
    
    # Price extraction
    price_patterns = [
        r'under ₹([\d\.]+)\s*(lakh|thousand)?',
        r'below ₹([\d\.]+)\s*(lakh|thousand)?',
        r'budget.*?₹([\d\.]+)\s*(lakh|thousand)?',
        r'₹([\d\.]+)\s*(lakh|thousand)?.*?budget'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            price_val = float(match.group(1))
            unit = match.group(2)
            if unit == 'lakh':
                criteria['max_price'] = price_val * 100000
            elif unit == 'thousand':
                criteria['max_price'] = price_val * 1000
            else:
                criteria['max_price'] = price_val
            break
    
    # Brand extraction
    brands = ['maruti', 'hyundai', 'honda', 'tata', 'datsun', 'toyota', 'mahindra']
    mentioned_brands = [brand for brand in brands if brand in user_input.lower()]
    if mentioned_brands:
        criteria['brands'] = mentioned_brands
    
    # Fuel type extraction
    if 'petrol' in user_input.lower():
        criteria['fuel_types'] = ['petrol']
    elif 'diesel' in user_input.lower():
        criteria['fuel_types'] = ['diesel']
    
    # Condition requirements
    if any(word in user_input.lower() for word in ['good condition', 'excellent', 'well maintained']):
        criteria['min_condition'] = 7.0
    elif 'low mileage' in user_input.lower():
        criteria['max_mileage'] = 50000
    
    return criteria

def build_car_buying_prompt(user_query: str, car_results: List[CarInfo], personality: str = "friendly") -> str:
    """Build contextual prompt for car buying assistant"""
    if not car_results:
        return f"""You are a helpful car buying assistant with a {personality} personality.
        
User Query: {user_query}
        
Unfortunately, no cars match the specified criteria. Please suggest:
1. Relaxing some requirements (budget, mileage, etc.)
2. Alternative brands or models to consider
3. Expanding the search area
        
Be encouraging and helpful in your response."""
    
    # Format car information
    car_context = "\n\n".join([f"Car {i+1}:\n{car.description}\n- Condition Score: {car.condition_score:.1f}/10\n- Estimated Discount: {car.discount_percentage:.1f}%" 
                              for i, car in enumerate(car_results)])
    
    personality_instructions = {
        "friendly": "Be warm, encouraging, and use emojis. Focus on helping the customer find their perfect car.",
        "professional": "Be formal, detailed, and focus on facts and specifications.",
        "casual": "Be relaxed, conversational, and use simple language."
    }
    
    prompt = f"""You are an expert car buying assistant with a {personality} personality.

<USER_QUERY>
{user_query}
</USER_QUERY>

<AVAILABLE_CARS>
{car_context}
</AVAILABLE_CARS>

<INSTRUCTIONS>
{personality_instructions.get(personality, personality_instructions['friendly'])}

</INSTRUCTIONS>

Provide a helpful, personalized recommendation based on the available cars."""
    
    return prompt

def generate_response(user_prompt: str) -> str:
    """Generate response using Azure OpenAI"""
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful car buying assistant that provides personalized recommendations."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

class CarBuyingAssistant:
    def __init__(self, csv_path: str):
        self.search_engine = CarSearchEngine(csv_path)
        self.personality = "friendly"  # Default personality
        self.conversation_history = []
        
    def set_personality(self, personality: str):
        """Set assistant personality: friendly, professional, or casual"""
        if personality in ["friendly", "professional", "casual"]:
            self.personality = personality
            print(f"Personality set to {personality}!")
        else:
            print("Invalid personality. Choose: friendly, professional, or casual")
    
    def search_cars(self, user_query: str) -> str:
        """Main search function with natural language processing"""
        # Extract criteria from user query
        criteria = extract_search_criteria(user_query)
        
        # Use first car's embedding as query vector (in real implementation, would embed the query)
        query_vector = self.search_engine.df["embedding"].iloc[0]
        
        # Search with criteria
        results = self.search_engine.hybrid_search(
            query_vector=query_vector,
            min_price=criteria.get('min_price'),
            max_price=criteria.get('max_price'),
            brands=criteria.get('brands'),
            fuel_types=criteria.get('fuel_types'),
            min_condition=criteria.get('min_condition'),
            max_mileage=criteria.get('max_mileage'),
            top_k=5
        )
        
        # Handle no results
        if not results:
            return self.handle_no_results(criteria)
        
        # Generate response
        prompt = build_car_buying_prompt(user_query, results, self.personality)
        response = generate_response(prompt)
        
        # Add formatted car details
        car_details = "\n\nDetailed Car Information:\n" + "\n\n".join([
            self.search_engine.format_car_result(car, include_trade_in=True) 
            for car in results
        ])
        
        return response + car_details
    
    def handle_no_results(self, criteria: Dict) -> str:
        """Handle case when no cars match criteria"""
        suggestions = []
        
        if 'max_price' in criteria:
            suggestions.append(f"- Consider increasing your budget above ₹{criteria['max_price']:,.0f}")
        if 'brands' in criteria:
            suggestions.append(f"- Try looking at other brands besides {', '.join(criteria['brands'])}")
        if 'min_condition' in criteria:
            suggestions.append("- Consider cars with slightly lower condition scores")
        if 'max_mileage' in criteria:
            suggestions.append(f"- Consider cars with mileage above {criteria['max_mileage']:,} km")
        
        fallback_msg = """No cars found matching your criteria!

Suggestions to find more options:
""" + "\n".join(suggestions) + """

Alternative approach:
- Try a broader search or ask me about specific brands/models
- I can also help you understand market trends and pricing

What would you like to adjust in your search?"""
        
        return fallback_msg
    
    def show_help(self):
        """Show help information"""
        print("""
Car Buying Assistant Help

What I can help with:
- Find cars by budget, brand, fuel type, condition
- Show price comparisons and discounts
- Estimate trade-in values
- Assess car condition based on mileage and ownership

Example queries:
- "I want a Maruti car under ₹2 lakh"
- "Show me diesel cars with good condition"
- "Find petrol cars with low mileage"
- "What's the best car for ₹5 lakh budget?"

Commands:
- /personality [friendly/professional/casual] - Change assistant style
- /help - Show this help
- /quit - Exit the assistant
""")
    
    def start_chat(self):
        """Start interactive chat interface"""
        print("Welcome to your AI Car Buying Assistant!")
        print("I can help you find the perfect car based on your needs.\n")
        print("Commands:")
        print("  - Ask about cars: 'I want a petrol car under ₹5 lakh'")
        print("  - Change personality: '/personality [friendly/professional/casual]'")
        print("  - Get help: '/help'")
        print("  - Quit: '/quit'")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.startswith('/quit'):
                    print("\nThank you for using the Car Buying Assistant! Happy car hunting!")
                    break
                elif user_input.startswith('/personality'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        self.set_personality(parts[1])
                    else:
                        print("Current personality:", self.personality)
                    continue
                elif user_input.startswith('/help'):
                    self.show_help()
                    continue
                
                # Process car search query
                print("\nAssistant: Searching for cars...\n")
                response = self.search_cars(user_input)
                print(response)
                print("\n" + "-"*60 + "\n")
                
                # Store conversation
                self.conversation_history.append((user_input, response))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    # Initialize and start the car buying assistant
    assistant = CarBuyingAssistant(CSV_PATH)
    assistant.start_chat()
