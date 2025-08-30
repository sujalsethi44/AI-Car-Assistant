# AI Car Buying Assistant

A comprehensive AI-powered car buying assistant that helps users find the perfect vehicle based on their specific requirements using advanced hybrid search capabilities and intelligent recommendations.

## Project Information

**Author**: Sujal Sethi  
**Email**: sujalsethi12344@gmail.com  
**Project Type**: AI Assignment - LLM + RAG Implementation

## Features

### Core Functionality
- **Intelligent Search**: Natural language query processing with vector similarity search
- **Advanced Filtering**: Filter by price, brand, fuel type, condition, and mileage
- **Condition Assessment**: 10-point scoring system based on vehicle age, mileage, and ownership
- **MSRP Discount Analysis**: Shows percentage savings from estimated original prices
- **Trade-in Estimates**: Calculates potential trade-in values for each vehicle
- **No Results Handling**: Intelligent suggestions when no cars match criteria

### Interface Options
- **Command Line Interface**: Interactive chat-based terminal application
- **Web Interface**: Modern Streamlit-based web application with intuitive UI
- **Personality Modes**: Choose between friendly, professional, or casual assistant styles

### Technical Features
- **Hybrid Search**: Combines vector similarity search with keyword filtering using FAISS
- **RAG Implementation**: Retrieval-Augmented Generation for contextual responses
- **Azure OpenAI Integration**: Powered by GPT-4 for intelligent recommendations
- **FAISS Vector Database**: Efficient similarity search across 4,340+ car listings
- **Pre-computed Embeddings**: Vector representations of car descriptions for semantic search
- **Vector Indexing**: FAISS IndexFlatL2 for fast similarity computations
- **Document Chunking**: Structured car data processing and embedding generation

## Screenshots

Application screenshots and output examples are available in the `Output/` folder:
- `cmd_ui_output.png` - Command line interface demonstration
- `web_ui_output.png` - Web interface demonstration
- `code_with_cmd_ui_output.png` - Code and CLI output combined view

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- Required Python packages (see requirements.txt)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials**:
   
   Edit the `.env` file and add your Azure OpenAI credentials:
   ```env
   API_BASE=your_azure_openai_endpoint
   API_KEY=your_azure_openai_api_key
   API_VERSION=2024-06-01
   DEPLOYMENT_NAME=your_deployment_name
   ```
   
   **Important**: You must provide valid Azure OpenAI credentials in the `.env` file for the application to function properly.

## Usage

### Command Line Interface

Run the interactive terminal-based assistant:

```bash
python car_assistant.py
```

**Available commands:**
- Ask about cars: "I want a petrol car under ₹5 lakh"
- Change personality: `/personality [friendly/professional/casual]`
- Get help: `/help`
- Quit: `/quit`

### Web Interface

Launch the web-based application:

```bash
streamlit run web_car_assistant.py
```

The web interface provides:
- Interactive sidebar filters for price, brand, fuel type, and condition
- Natural language search box
- Visual car cards with detailed information
- Progress bars for condition scores and discounts

## Example Queries

- "I want a Maruti car under ₹2 lakh"
- "Show me diesel cars with good condition"
- "Find petrol cars with low mileage"
- "What's the best car for ₹5 lakh budget?"
- "I need a reliable family car under ₹4 lakh"

## File Structure

```
├── car_assistant.py          # Command line interface
├── web_car_assistant.py      # Streamlit web interface
├── hybrid_search.py          # Legacy search implementation
├── car_descriptions_embeddings.csv  # Car data with embeddings
├── car_details_nlp.txt       # Processed car descriptions
├── requirements.txt          # Python dependencies
├── .env                      # API credentials (configure this!)
├── Output/                   # Screenshots and examples
│   ├── cmd_ui_output.png
│   ├── web_ui_output.png
│   └── code_with_cmd_ui_output.png
└── README.md                 # This file
```

## Data

The system includes a comprehensive dataset of 4,340+ car listings with:
- Vehicle specifications (brand, model, year, price)
- Condition information (mileage, ownership history)
- Pre-computed embeddings for vector search
- Estimated MSRP and discount calculations

## Approach & Methodology

### Problem-Solving Strategy
This project implements a comprehensive car buying assistant using a **Retrieval-Augmented Generation (RAG)** approach combined with **hybrid search** capabilities. The solution addresses the challenge of helping users find suitable cars from a large dataset through intelligent filtering and natural language interaction.

### Implementation Approach

#### 1. **Data Preparation & Embedding Generation**
- **Dataset Processing**: 4,340+ car descriptions converted to structured format
- **Text Preprocessing**: Car descriptions cleaned and standardized for consistency
- **Embedding Creation**: Pre-computed vector embeddings using sentence transformers
- **Data Storage**: Embeddings stored alongside original data in CSV format for efficient access

#### 2. **Vector Search Implementation**
- **FAISS Integration**: Facebook AI Similarity Search for high-performance vector operations
- **Index Creation**: L2 distance-based similarity index for semantic search
- **Chunking Strategy**: Car descriptions treated as individual searchable documents
- **Similarity Matching**: Cosine similarity for finding semantically related vehicles

#### 3. **Hybrid Search Architecture**
- **Vector + Keyword Combination**: Semantic search enhanced with traditional filtering
- **Multi-criteria Filtering**: Price, brand, fuel type, condition, mileage constraints
- **Ranking Algorithm**: Combined scoring based on similarity and filter relevance
- **Real-time Processing**: Dynamic query processing with instant results

#### 4. **RAG Pipeline Design**
- **Retrieval Phase**: FAISS-based similarity search to find relevant cars
- **Context Building**: Retrieved car data formatted for LLM consumption
- **Augmentation**: User query combined with retrieved car information
- **Generation**: Azure OpenAI GPT-4 generates personalized recommendations

#### 5. **User Interface Strategy**
- **Dual Interface Approach**: Both CLI and web-based interfaces for different use cases
- **Natural Language Processing**: Query parsing to extract search criteria
- **Interactive Filtering**: Real-time filter application with immediate feedback
- **Progressive Enhancement**: Basic functionality with advanced features layered on top

#### 6. **Business Logic Integration**
- **Condition Assessment**: Multi-factor scoring algorithm for vehicle condition
- **Price Analysis**: MSRP estimation and discount calculation logic
- **Market Intelligence**: Trade-in value estimation based on depreciation models
- **Fallback Handling**: Intelligent suggestions when no matches found

### Key Design Decisions

#### **Why RAG Over Fine-tuning?**
- **Data Freshness**: Easy to update car inventory without retraining
- **Cost Efficiency**: No expensive model fine-tuning required
- **Flexibility**: Can adapt to different car markets and datasets
- **Transparency**: Clear traceability of recommendations to source data

#### **Why FAISS Over Traditional Search?**
- **Semantic Understanding**: Captures meaning beyond keyword matching
- **Performance**: Sub-millisecond search across thousands of records
- **Scalability**: Easily handles growing datasets
- **Accuracy**: Better matching of user intent to available cars

#### **Why Hybrid Search?**
- **Best of Both Worlds**: Combines semantic similarity with exact filtering
- **User Control**: Allows precise constraints (budget, brand) with flexible matching
- **Relevance**: Ensures results meet both semantic and practical requirements
- **Fallback Options**: Graceful degradation when strict filters yield no results

## Technical Architecture

### Search Engine
- **FAISS Vector Database**: Facebook AI Similarity Search for efficient nearest neighbor search
- **Vector Embeddings**: Pre-computed 768-dimensional embeddings for car descriptions
- **Index Structure**: FAISS IndexFlatL2 for L2 distance-based similarity search
- **Hybrid Search**: Combines vector similarity with keyword-based filtering
- **Document Chunking**: Car descriptions processed into structured chunks for embedding
- **Multi-criteria Filtering**: Price, brand, fuel type, condition, mileage filtering

### AI Integration
- **Azure OpenAI GPT-4**: Large Language Model for natural language understanding
- **RAG Pipeline**: Retrieval-Augmented Generation combining search results with LLM
- **Embedding Generation**: Sentence transformers for semantic vector representations
- **Prompt Engineering**: Contextual prompts with personality customization
- **Response Generation**: Contextual car recommendations based on retrieved data

### Data Processing & Storage
- **CSV Data Management**: Structured car data with embedded vectors
- **Condition Scoring Algorithm**: Multi-factor scoring (age, mileage, ownership)
- **MSRP Estimation Engine**: Brand/model-based original price calculations
- **Trade-in Value Calculator**: Market value-based depreciation estimates
- **Real-time Filtering**: Dynamic query processing and result ranking

## Configuration

### Environment Variables
All API credentials are managed through the `.env` file:
- `API_BASE`: Azure OpenAI endpoint URL
- `API_KEY`: Azure OpenAI API key
- `API_VERSION`: API version 
- `DEPLOYMENT_NAME`: Your GPT-4 deployment name

### Personality Modes
- **Friendly**: Warm, encouraging responses with helpful tone
- **Professional**: Formal, detailed, fact-focused responses
- **Casual**: Relaxed, conversational, simple language

## Troubleshooting

### Common Issues

1. **API Authentication Error**
   - Verify your `.env` file contains valid Azure OpenAI credentials
   - Check that your API key has proper permissions

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt` to install all required packages

3. **Data Loading Issues**
   - Ensure `car_descriptions_embeddings.csv` is in the same directory
   - Check file permissions and encoding

4. **Streamlit Port Issues**
   - Use `streamlit run web_car_assistant.py --server.port 8502` to specify a different port

## Assignment Requirements Compliance

This implementation fully satisfies all assignment requirements:

✅ **LLM Selection & Adaptation**: Azure OpenAI GPT-4 with RAG strategy  
✅ **Data Integration**: 4,340+ car listings with structured data  
✅ **Interactive Interface**: Both command-line and web-based chat interfaces  
✅ **Advanced Filtering**: Price, brand, condition, fuel type, mileage filtering  
✅ **MSRP Discount Display**: Percentage savings calculations  
✅ **No Results Handling**: Intelligent fallback suggestions  
✅ **Bonus Features**: Trade-in estimates, personality customization, professional UI  

