from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import functions.kmedoids as kmedoids
import functions.DTW as dtw
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform

def clustering_with_dtw(X, n_components):
    """
    Performs clustering using Dynamic Time Warping (DTW) distance and K-Medoids.

    Parameters:
    X (array-like): Input data.
    n_components (int): Number of clusters.

    Returns:
    tuple: Medoids, labels, and loss of the clustering.
    """
    # Compute DTW distance matrix
    dtw_matrix = dtw.compute_dtw(X)
    print(X.shape)

    # Initialize and train K-Medoids with DTW distance
    KMedoid = kmedoids.KMedoids(K=n_components)
    medoids, labels, loss = KMedoid.train(dtw_matrix, init='kmedoids++', random_state=1)

    return medoids, labels

def clustering_with_euclidean(X, n_components):
    """
    Performs clustering using Euclidean distance and K-Medoids.

    Parameters:
    X (array-like): Input data.
    n_components (int): Number of clusters.

    Returns:
    tuple: Medoids, labels, and loss of the clustering.
    """
    # Compute Euclidean distance matrix
    euclidean_matrix = distance_matrix(x=X, y=X, p=2)
    print("Euclidean matrix: ", euclidean_matrix.shape)
    print(euclidean_matrix[:4, :4])

    # Initialize and train K-Medoids with Euclidean distance
    KMedoid = kmedoids.KMedoids(K=n_components)
    medoids, labels, loss = KMedoid.train(euclidean_matrix, init='kmedoids++', random_state=1)

    return medoids, labels

def hierarchical_clustering(X):
    """
    Performs hierarchical clustering using DTW distance.

    Parameters:
    X (array-like): Input data.
    """
    # Compute DTW distance matrix
    dtw_matrix = dtw.compute_dtw(X)
    dist_vector = squareform(dtw_matrix)

    # Compute linkage matrix and plot dendrogram
    linkage_matrix = linkage(dist_vector, method='ward')
    plt.figure()
    dendrogram(linkage_matrix)
    plt.show()

def k_means(X, n_clusters):
    """
    Performs clustering using K-Means.

    Parameters:
    X (array-like): Input data.
    n_clusters (int): Number of clusters.

    Returns:
    array: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    return clusters
