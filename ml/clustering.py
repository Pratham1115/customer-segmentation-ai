from sklearn.cluster import KMeans

def perform_clustering(data, n_clusters):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    clusters = model.fit_predict(data)

    return clusters
