import streamlit as st
import pandas as pd

from ml.preprocess import clean_data
from ml.scaling import scale_features
from ml.clustering import perform_clustering

# Page config
st.set_page_config(
    page_title="AI Customer Segmentation",
    layout="wide"
)

# Sidebar
st.sidebar.title("AI Customer Dashboard")

# Main title
st.title("AI-Powered Customer Segmentation")

st.write("""
Analyze customer behavior using Machine Learning.
""")

# Upload file
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Original dataset
    st.subheader("Original Dataset")

    st.dataframe(df)

    # Clean data
    df = clean_data(df)

    # Cleaned dataset
    st.subheader("Cleaned Dataset")

    st.dataframe(df)

    # Numeric columns only
    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # Feature selection
    st.subheader("Select Features")

    selected_features = st.multiselect(
        "Choose columns for clustering",
        numeric_columns,
        default=numeric_columns[1:4]
    )

    if len(selected_features) > 0:

        # Selected data
        selected_data = df[selected_features]

        # Scaling
        scaled_data = scale_features(selected_data)

        st.success("Feature scaling completed!")

        # Cluster slider
        n_clusters = st.slider(
            "Select Number of Clusters",
            2,
            10,
            3
        )

        # Perform clustering
        clusters = perform_clustering(
            scaled_data,
            n_clusters
        )

        # Add cluster column
        df["Cluster"] = clusters

        # Show result
        st.subheader("Clustered Dataset")

        st.dataframe(df)

        # Cluster counts
        st.subheader("Cluster Distribution")

        st.write(df["Cluster"].value_counts())