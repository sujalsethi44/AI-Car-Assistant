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
COLLECTION_NAME = "car_descriptions"

"""
Update: Handles CSV with columns 'car_description' and 'embedding'.
Uses row index as unique ID.
"""

# Load CSV (expects columns: car_description, embedding)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure embedding is a list of floats
    df["embedding"] = df["embedding"].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])
    return df

# Initialize Faiss index
def init_faiss(vector_size):
    index = faiss.IndexFlatL2(vector_size)
    return index

# Upload data to Faiss
def upload_to_faiss(index, df):
    embeddings = df["embedding"].tolist()
    index.add(np.array(embeddings, dtype="float32"))
    descriptions = df["car_description"].tolist()
    return descriptions

# Hybrid search: vector + keyword
def hybrid_search(index, descriptions, query_vector, keyword=None, top_k=5):
    query = np.array([query_vector], dtype="float32")
    D, I = index.search(query, top_k*2)  # get more results for post-filtering
    results = [(descriptions[i], D[0][idx]) for idx, i in enumerate(I[0])]
    if keyword:
        filtered = [(desc, score) for desc, score in results if keyword.lower() in desc.lower()]
        return filtered[:top_k]
    return results[:top_k]

import numpy as np

def build_contextual_prompt(user_query: str, chunks: list[str]) -> str:
    """
    Constructs a detailed, contextual prompt tailored to Revenue Cycle Management (RCM) support tasks,
    using reranked internal documentation chunks and a user query.

    Parameters:
        user_query (str): The user's instruction or question related to medical claims processing.
        chunks (list[str]): Top-N reranked content chunks from internal SOPs, payer guides, or system notes.

    Returns:
        str: A formatted prompt suitable for language model reasoning in the context of RCM operations.
    """
    # Join all cleaned chunks with separators
    context = "\n---\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    prompt = f""" You are expert engine that answers questions using provided context.

<INPUT>
- User query: {user_query}
</INPUT>

<TASK>
- Use the provided context below to generate response.
</TASK>

<CONTEXT>
---------------------
{context}
---------------------
</CONTEXT>

<INSTRUCTIONS>
- If the user asks a greeting or casual question, respond accordingly but maintain focus on the RCM context.
- **Only provide instructions explicitly stated in the context; do not generate or infer additional suggestions or instructions not present in the context.**
</INSTRUCTIONS>

<OUTPUT>
- Provide only the information explicitly requested in the user query.
</OUTPUT>
"""

    return prompt

def generate_response(user_prompt):
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages = [
                {"role": "system", "content": "You are a helpful assistant that responds based on the provided prompt"},
                {"role": "user", "content": f"{user_prompt}"}],
            max_tokens=1024,
            temperature=0,
            )
    except Exception as e:
        print("Error:", str(e))

    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    df = load_data(CSV_PATH)
    vector_size = len(df["embedding"].iloc[0])
    index = init_faiss(vector_size)
    descriptions = upload_to_faiss(index, df)
    # Example search
    example_vector = df["embedding"].iloc[0]
    results = hybrid_search(index, descriptions, example_vector, keyword="maruti", top_k=5)
    
    retrieved_chunks = []
    for desc, score in results:
        retrieved_chunks.append(desc)
        retrieved_chunks = list(set(retrieved_chunks))

    prompt = build_contextual_prompt(user_query="Which car should I buy? It should be a petrol-powered vehicle with manual transmission, being sold by an individual seller for ₹60,000", chunks=retrieved_chunks)
    # print(prompt)
    #generating final response
    res = generate_response(prompt)
    print("Final response:")
    print(res)