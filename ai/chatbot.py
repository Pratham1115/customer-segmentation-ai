import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def render_chatbot():
    st.title("🤖 AI Analytics Chatbot")
    st.caption("Ask me anything about your customer segments and business data!")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("API Key not found. Please check your .env file.")
        st.stop()

    genai.configure(api_key=api_key)

    # Give the AI context about the dataset if the user has already uploaded and clustered it
    system_instruction = "You are an expert AI business analytics assistant helping a user understand their customer segmentation dashboard."
    
    if "processed_data" in st.session_state:
        df = st.session_state.processed_data
        if 'Cluster' in df.columns:
            # Get the summary of the data so the chatbot can reference it
            num_cols = [col for col in df.select_dtypes(include=['number']).columns if not col.startswith('PCA')]
            summary = df.groupby('Cluster')[num_cols].mean().round(2).to_dict()
            system_instruction += f" Here is the current mathematical summary of the user's customer clusters: {summary}. Use this exact data to answer their questions accurately."

    # Initialize the model with the system instructions
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=system_instruction
    )

    # Initialize chat session in memory
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display chat history
    for message in st.session_state.chat_session.history:
        role = "assistant" if message.role == "model" else message.role
        with st.chat_message(role):
            st.markdown(message.parts[0].text)

    # Handle user input
    if prompt := st.chat_input("E.g., Which cluster spends the most?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.chat_session.send_message(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"An API error occurred: {e}")