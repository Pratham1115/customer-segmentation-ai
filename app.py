import streamlit as st
import pandas as pd

from ml.preprocess import clean_data
from ml.scaling import scale_features
from ml.clustering import perform_clustering
from ml.pca_visual import reduce_dimensions

from visuals.charts import (
    cluster_scatter_plot,
    cluster_pie_chart
)

# Page config
st.set_page_config(
    page_title="AI Customer Segmentation",
    layout="wide"
)

# Sidebar
st.sidebar.title("AI Customer Dashboard")

st.sidebar.info("""
Upload customer datasets,
perform machine learning,
and visualize customer behavior.
""")

# Main title
st.title("AI-Powered Customer Segmentation")

st.write("""
Analyze customer behavior using Machine Learning
and interactive analytics.
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Original data
    st.subheader("Dataset Preview")

    st.dataframe(df)

    # Clean data
    df = clean_data(df)

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # Feature selection
    st.subheader("Feature Selection")

    selected_features = st.multiselect(
        "Select columns for clustering",
        numeric_columns,
        default=numeric_columns[1:4]
    )

    if len(selected_features) > 0:

        # Selected data
        selected_data = df[selected_features]

        # Scale data
        scaled_data = scale_features(selected_data)

        st.success("Feature scaling completed!")

        # Cluster slider
        n_clusters = st.slider(
            "Select Number of Clusters",
            2,
            10,
            3
        )

        # K-Means clustering
        clusters = perform_clustering(
            scaled_data,
            n_clusters
        )

        # Add cluster labels
        df["Cluster"] = clusters

        # PCA reduction
        reduced_data = reduce_dimensions(
            scaled_data
        )

        # Cluster distribution
        cluster_counts = df["Cluster"].value_counts()

        # Show dataframe
        st.subheader("Clustered Dataset")

        st.dataframe(df)

        # Metrics
        st.subheader("Dataset Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Customers",
                len(df)
            )

        with col2:
            st.metric(
                "Features Used",
                len(selected_features)
            )

        with col3:
            st.metric(
                "Clusters",
                n_clusters
            )

        # Scatter plot
        st.subheader("Customer Segments Visualization")

        scatter_fig = cluster_scatter_plot(
            reduced_data,
            clusters
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )

        # Pie chart
        st.subheader("Cluster Distribution")

        pie_fig = cluster_pie_chart(
            cluster_counts
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )