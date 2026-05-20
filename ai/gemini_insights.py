import google.generativeai as genai
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_insights(cleaned_data):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Error: API Key not found. Please check your .env file."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Ensure the data has been clustered
    if 'Cluster' not in cleaned_data.columns:
        return "⚠️ Error: Data has not been clustered yet. Please run the clustering step first."
        
    # Calculate the average values for each cluster to send to the AI
    # We ignore the PCA columns so we don't confuse the model with coordinates
    numeric_cols = cleaned_data.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if not col.startswith('PCA')]
    
    cluster_summary = cleaned_data.groupby('Cluster')[numeric_cols].mean().round(2).to_dict()
    cluster_sizes = cleaned_data['Cluster'].value_counts().to_dict()

    # Build the prompt dynamically using the real data
    prompt = f"""
    You are an expert business analyst and marketing strategist.
    I have segmented my customers into {len(cluster_sizes)} clusters using K-Means.
    
    Here is the average data for each customer cluster:
    {cluster_summary}
    
    Here is the number of customers in each cluster:
    {cluster_sizes}
    
    Based on this data, please provide:
    1. A short, catchy persona name for each cluster (e.g., "Premium Spenders", "Budget Shoppers").
    2. A brief analysis of their behavior.
    3. Specific, actionable marketing recommendations for each group.
    
    Format the output cleanly using Markdown headers and bullet points.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An API error occurred: {e}"