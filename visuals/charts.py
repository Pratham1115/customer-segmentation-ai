import pandas as pd
import plotly.express as px

# Scatter plot
def cluster_scatter_plot(reduced_data, clusters):

    plot_df = pd.DataFrame({
        "PCA1": reduced_data[:, 0],
        "PCA2": reduced_data[:, 1],
        "Cluster": clusters.astype(str)
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="Customer Segments",
        width=1000,
        height=600
    )

    return fig

# Pie chart
def cluster_pie_chart(cluster_counts):

    fig = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index.astype(str),
        title="Cluster Distribution"
    )

    return fig
