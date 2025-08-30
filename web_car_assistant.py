import streamlit as st
import pandas as pd
import faiss
import numpy as np
import openai
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Car Buying Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE = os.getenv('API_BASE')
API_KEY = os.getenv('API_KEY')
API_VERSION = os.getenv('API_VERSION')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION

CSV_PATH = "car_descriptions_embeddings.csv"

@dataclass
class CarInfo:
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

@st.cache_data
def load_car_data():
    """Load and cache car data"""
    df = pd.read_csv(CSV_PATH)
    df["embedding"] = df["embedding"].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])
    
    car_infos = []
    for _, row in df.iterrows():
        car_info = parse_car_description(row['car_description'])
        car_infos.append(car_info)
    
    return df, car_infos

def parse_car_description(description: str) -> CarInfo:
    """Extract structured information from car description"""
    year_match = re.search(r'(\d{4}) model', description)
    year = int(year_match.group(1)) if year_match else 2010
    
    brand_model_match = re.search(r'model ([A-Za-z]+)\s+([A-Za-z0-9\s\.\-]+?)\s+is', description)
    if brand_model_match:
        brand = brand_model_match.group(1)
        model = brand_model_match.group(2).strip()
    else:
        brand = "Unknown"
        model = "Unknown"
    
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
    
    fuel_type = "petrol" if "petrol-powered" in description else "diesel" if "diesel-powered" in description else "unknown"
    transmission = "manual" if "manual transmission" in description else "automatic" if "automatic transmission" in description else "unknown"
    
    mileage_match = re.search(r'driven for ([\d,]+)\s*km', description)
    mileage = int(mileage_match.group(1).replace(',', '')) if mileage_match else 0
    
    ownership_match = re.search(r'(first|second|third) owner', description)
    ownership = ownership_match.group(1) if ownership_match else "unknown"
    
    condition_score = calculate_condition_score(year, mileage, ownership)
    estimated_msrp = estimate_msrp(brand, model, year)
    discount_percentage = ((estimated_msrp - price) / estimated_msrp * 100) if estimated_msrp > 0 else 0
    
    return CarInfo(
        description=description, brand=brand, model=model, year=year, price=price,
        fuel_type=fuel_type, transmission=transmission, mileage=mileage,
        ownership=ownership, condition_score=condition_score,
        estimated_msrp=estimated_msrp, discount_percentage=discount_percentage
    )

def calculate_condition_score(year: int, mileage: int, ownership: str) -> float:
    current_year = 2024
    age = current_year - year
    score = 10.0
    score -= age * 0.5
    score -= (mileage / 20000) * 1.0
    if ownership == "second":
        score -= 1.0
    elif ownership == "third":
        score -= 2.0
    return max(0, min(10, score))

def estimate_msrp(brand: str, model: str, year: int) -> float:
    base_prices = {
        "maruti": {"800": 250000, "alto": 300000, "wagon r": 400000, "swift": 500000, "baleno": 600000},
        "hyundai": {"verna": 800000, "xcent": 600000, "creta": 1000000, "i10": 400000, "i20": 600000},
        "honda": {"amaze": 600000, "city": 900000, "jazz": 700000},
        "tata": {"indigo": 500000, "nano": 200000, "bolt": 600000},
        "datsun": {"redigo": 350000, "go": 400000}
    }
    
    brand_lower = brand.lower()
    model_lower = model.lower()
    base_price = 500000
    
    if brand_lower in base_prices:
        for model_key, price in base_prices[brand_lower].items():
            if model_key in model_lower:
                base_price = price
                break
    
    current_year = 2024
    age = current_year - year
    return base_price * (1 + age * 0.05)

@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the search engine"""
    df, car_infos = load_car_data()
    vector_size = len(df["embedding"].iloc[0])
    index = faiss.IndexFlatL2(vector_size)
    embeddings = df["embedding"].tolist()
    index.add(np.array(embeddings, dtype="float32"))
    return df, car_infos, index

def filter_cars(car_infos: List[CarInfo], **criteria) -> List[CarInfo]:
    """Filter cars by various criteria"""
    filtered = car_infos.copy()
    
    if criteria.get('min_price'):
        filtered = [car for car in filtered if car.price >= criteria['min_price']]
    if criteria.get('max_price'):
        filtered = [car for car in filtered if car.price <= criteria['max_price']]
    if criteria.get('brands'):
        filtered = [car for car in filtered if car.brand.lower() in [b.lower() for b in criteria['brands']]]
    if criteria.get('fuel_types'):
        filtered = [car for car in filtered if car.fuel_type.lower() in [f.lower() for f in criteria['fuel_types']]]
    if criteria.get('min_condition'):
        filtered = [car for car in filtered if car.condition_score >= criteria['min_condition']]
    if criteria.get('max_mileage'):
        filtered = [car for car in filtered if car.mileage <= criteria['max_mileage']]
        
    return filtered

def search_cars(df, car_infos, index, query_vector, **criteria) -> List[CarInfo]:
    """Enhanced hybrid search with filtering"""
    query = np.array([query_vector], dtype="float32")
    D, I = index.search(query, len(car_infos))
    
    search_results = [(car_infos[i], D[0][idx]) for idx, i in enumerate(I[0])]
    filtered_cars = [car for car, _ in search_results]
    filtered_cars = filter_cars(filtered_cars, **criteria)
    
    final_results = []
    for car, score in search_results:
        if car in filtered_cars:
            final_results.append((car, score))
            if len(final_results) >= criteria.get('top_k', 5):
                break
    
    return [car for car, _ in final_results]

def get_trade_in_estimate(car_info: CarInfo) -> float:
    """Estimate trade-in value"""
    market_value = car_info.price
    trade_in_percentage = 0.75 - (car_info.condition_score / 100)
    return market_value * max(0.6, trade_in_percentage)

def generate_ai_response(user_query: str, car_results: List[CarInfo], personality: str) -> str:
    """Generate AI response using Azure OpenAI"""
    if not car_results:
        prompt = f"""You are a helpful car buying assistant with a {personality} personality.
        
User Query: {user_query}
        
No cars match the criteria. Suggest alternatives and be encouraging."""
    else:
        car_context = "\n\n".join([f"Car {i+1}: {car.brand} {car.model} ({car.year}) - ₹{car.price:,.0f}, {car.fuel_type}, {car.mileage:,}km, {car.ownership} owner, Condition: {car.condition_score:.1f}/10" 
                                  for i, car in enumerate(car_results)])
        
        prompt = f"""You are a car buying assistant with {personality} personality.

User Query: {user_query}

Available Cars:
{car_context}

Provide personalized recommendations explaining why each car fits their needs."""
    
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful car buying assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Initialize data
df, car_infos, index = initialize_search_engine()

# Main UI
st.title("AI Car Buying Assistant")
st.markdown("Find your perfect car with intelligent search and personalized recommendations!")

# Sidebar for filters
st.sidebar.header("Search Filters")

# Personality selection
personality = st.sidebar.selectbox(
    "Assistant Personality",
    ["friendly", "professional", "casual"],
    help="Choose how the assistant should communicate"
)

# Price filter
st.sidebar.subheader("Budget")
price_range = st.sidebar.slider(
    "Price Range (₹)",
    min_value=0,
    max_value=1000000,
    value=(0, 1000000),
    step=50000,
    format="₹%d"
)

# Brand filter
available_brands = list(set([car.brand for car in car_infos]))
selected_brands = st.sidebar.multiselect(
    "Brands",
    available_brands,
    help="Select specific brands (leave empty for all)"
)

# Fuel type filter
fuel_types = st.sidebar.multiselect(
    "Fuel Type",
    ["petrol", "diesel"],
    help="Select fuel types"
)

# Condition filter
min_condition = st.sidebar.slider(
    "Minimum Condition Score",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Filter by condition score (0-10)"
)

# Mileage filter
max_mileage = st.sidebar.number_input(
    "Maximum Mileage (km)",
    min_value=0,
    max_value=500000,
    value=500000,
    step=10000,
    help="Maximum acceptable mileage"
)

# Main search area
user_query = st.text_area(
    "What kind of car are you looking for?",
    placeholder="e.g., I want a reliable petrol car under ₹3 lakh for city driving",
    height=100
)

# Search button
if st.button("Search Cars", type="primary"):
    if user_query:
        with st.spinner("Searching for cars..."):
            # Prepare search criteria
            criteria = {
                'min_price': price_range[0] if price_range[0] > 0 else None,
                'max_price': price_range[1] if price_range[1] < 1000000 else None,
                'brands': selected_brands if selected_brands else None,
                'fuel_types': fuel_types if fuel_types else None,
                'min_condition': min_condition if min_condition > 0 else None,
                'max_mileage': max_mileage if max_mileage < 500000 else None,
                'top_k': 5
            }
            
            # Use first car's embedding as query vector
            query_vector = df["embedding"].iloc[0]
            
            # Search cars
            results = search_cars(df, car_infos, index, query_vector, **criteria)
            
            if results:
                # Generate AI response
                ai_response = generate_ai_response(user_query, results, personality)
                
                # Display AI response
                st.subheader("AI Recommendation")
                st.markdown(ai_response)
                
                # Display car results
                st.subheader("Found Cars")
                
                for i, car in enumerate(results):
                    with st.expander(f"{car.brand} {car.model} ({car.year}) - ₹{car.price:,.0f}", expanded=i==0):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **Basic Info:**
                            - Brand: {car.brand}
                            - Model: {car.model}
                            - Year: {car.year}
                            - Price: ₹{car.price:,.0f}
                            - Fuel: {car.fuel_type.title()}
                            - Transmission: {car.transmission.title()}
                            """)
                        
                        with col2:
                            condition_text = "Excellent" if car.condition_score >= 8 else "Good" if car.condition_score >= 6 else "Fair" if car.condition_score >= 4 else "Poor"
                            trade_in = get_trade_in_estimate(car)
                            
                            st.markdown(f"""
                            **Condition & Value:**
                            - Mileage: {car.mileage:,} km
                            - Ownership: {car.ownership.title()} owner
                            - Condition: {condition_text} ({car.condition_score:.1f}/10)
                            - Discount: {car.discount_percentage:.1f}% off MSRP
                            - Trade-in Est: ₹{trade_in:,.0f}
                            """)
                        
                        # Progress bars for visual appeal
                        st.markdown("**Condition Score:**")
                        st.progress(car.condition_score / 10)
                        
                        if car.discount_percentage > 0:
                            st.markdown("**Discount Percentage:**")
                            st.progress(min(car.discount_percentage / 100, 1.0))
            else:
                st.error("No cars found matching your criteria!")
                st.markdown("""
                **Suggestions:**
                - Try increasing your budget
                - Consider different brands
                - Relax condition requirements
                - Increase maximum mileage
                """)
    else:
        st.warning("Please enter your car requirements to search!")


# Handle quick filters
if hasattr(st.session_state, 'quick_filter'):
    if st.session_state.quick_filter == "best_condition":
        filtered_cars = [car for car in car_infos if car.condition_score >= 8]
    elif st.session_state.quick_filter == "budget":
        filtered_cars = [car for car in car_infos if car.price <= 200000]
    elif st.session_state.quick_filter == "petrol":
        filtered_cars = [car for car in car_infos if car.fuel_type == "petrol"]
    elif st.session_state.quick_filter == "low_mileage":
        filtered_cars = [car for car in car_infos if car.mileage <= 50000]
    
    if filtered_cars:
        st.subheader(f"Quick Filter Results ({len(filtered_cars)} cars)")
        for car in filtered_cars[:5]:  # Show top 5
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{car.brand} {car.model} ({car.year})**")
                with col2:
                    st.write(f"₹{car.price:,.0f}")
                with col3:
                    st.write(f"{car.condition_score:.1f}/10")
    
    # Clear the filter
    del st.session_state.quick_filter

# Chat history in sidebar
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history:
    st.sidebar.header("Recent Searches")
    for i, (query, _) in enumerate(st.session_state.chat_history[-3:]):  # Show last 3
        st.sidebar.text(f"{i+1}. {query[:30]}...")
