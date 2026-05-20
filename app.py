import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- IMPORTING YOUR MODULES ---
# These imports match the folder structure defined in your PRD.
# Ensure your functions in these files are named accordingly.
try:
    from ml.preprocess import clean_data
    from ml.scaling import scale_features
    from ml.clustering import perform_clustering
    from ml.pca_visual import reduce_dimensions
    from ai.chatbot import render_chatbot
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}. Please ensure your folder structure and function names match the PRD.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Customer Segmentation", page_icon="📊", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "AI Chatbot"])

if menu == "Dashboard":
    st.title("📊 AI-Powered Customer Segmentation")
    st.write("Upload your customer dataset to clean, segment, and visualize the data.")

    # FEATURE 1: CSV Dataset Upload
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Read data
        raw_data = pd.read_csv(uploaded_file)
        
        # FEATURE 2: Dataset Preview
        st.subheader("1. Dataset Preview")
        st.write(f"Rows: {raw_data.shape[0]} | Columns: {raw_data.shape[1]}")
        st.dataframe(raw_data.head())

        # FEATURE 3: Data Cleaning
        st.subheader("2. Data Cleaning Pipeline")
        cleaned_data = clean_data(raw_data)
        st.success(f"Data cleaned! Remaining rows: {cleaned_data.shape[0]}")

        # FEATURE 4: Feature Selection
        st.subheader("3. Feature Selection for Clustering")
        numeric_columns = cleaned_data.select_dtypes(include=['number']).columns.tolist()
        
        selected_features = st.multiselect(
            "Select at least two numerical features to cluster by:",
            options=numeric_columns,
            default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
        )

        if len(selected_features) >= 2:
            clustering_data = cleaned_data[selected_features]

            # FEATURE 5: Feature Scaling
            st.subheader("4. Feature Scaling")
            scaled_data = scale_features(clustering_data)
            st.write("Data normalized using StandardScaler.")

            # FEATURE 6: Customer Segmentation (K-Means)
            st.subheader("5. Customer Segmentation")
            n_clusters = st.slider("Select number of customer clusters:", min_value=2, max_value=10, value=4)
            
            # Perform clustering and add labels back to the dataframe
            cluster_labels = perform_clustering(scaled_data, n_clusters)
            cleaned_data['Cluster'] = cluster_labels
            cleaned_data['Cluster'] = cleaned_data['Cluster'].astype(str) # String for categorical coloring
            
            st.write("Customer segments generated successfully.")

            # FEATURE 7 & 8: PCA Reduction and Interactive Visualization
            st.subheader("6. Interactive Visualization")
            
            # Reduce to 2D for the scatter plot
            pca_data = reduce_dimensions(scaled_data)
            cleaned_data['PCA1'] = pca_data[:, 0]
            cleaned_data['PCA2'] = pca_data[:, 1]

            col1, col2 = st.columns(2)
            
            with col1:
                # PCA Scatter Plot
                fig_scatter = px.scatter(
                    cleaned_data, 
                    x='PCA1', 
                    y='PCA2', 
                    color='Cluster',
                    title="Customer Segments (2D PCA View)",
                    hover_data=selected_features
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                # Cluster Distribution Pie Chart
                cluster_counts = cleaned_data['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig_pie = px.pie(
                    cluster_counts, 
                    names='Cluster', 
                    values='Count',
                    title="Cluster Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # FEATURE 9: AI Business Insights
            st.subheader("7. AI Business Insights")
            if st.button("Generate AI Insights"):
                with st.spinner("Analyzing cluster data with Gemini..."):
                    from ai.gemini_insights import generate_insights
                    insights = generate_insights(cleaned_data)
                    st.markdown(insights)
                    
            # Save processed data to session state so the chatbot can access it
            st.session_state.processed_data = cleaned_data

        else:
            st.warning("Please select at least two features to perform clustering.")

elif menu == "AI Chatbot":
    # FEATURE 10: AI Analytics Chatbot
    render_chatbot()