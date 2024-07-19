import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import plotly.graph_objs as go

def show_curve(curve, title=""):
    """
    Plots a set of curves on a 2D graph.

    Parameters:
    curve (array-like): List of curves to plot.
    title (str): Title of the plot.
    """
    curve = np.array(curve)
    plt.figure(figsize=(10, 6))

    # Use a colormap for the curves
    colormap = cm.get_cmap('jet', curve.shape[0])

    for i, single_curve in enumerate(curve):
        plt.plot(single_curve, linestyle='-', color=colormap(i), label=f'Curve {i + 1}')

    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Amplitude')
    if len(curve) < 20:
        plt.legend()
    plt.show()

def show_fill_curve(medoids, n_components, curves, nb_label, color='yellow'):
    """
    Plots and fills between curves on a 2D graph.

    Parameters:
    medoids (array-like): Medoid curves to plot.
    n_components (int): Number of components (curves) to plot.
    curves (array-like): Curves to fill between.
    nb_label (list): Labels for the curves.
    color (str): Fill color between the curves.
    """
    print(nb_label)
    title = [f' Curve {i + 1}: {nb_label[i]}  ' for i in range(n_components)]
    lengths = [len(curve) for curve in curves]
    if len(set(lengths)) != 1:
        raise ValueError("All curves must have the same length")

    plt.figure(figsize=(10, 6))
    x = np.arange(lengths[0])

    # Plot medoids
    for i, curve in enumerate(medoids):
        plt.plot(x, curve, label=f'Curve {i + 1}')

    # Fill curves 2 by 2
    for i in range(len(curves) - 1):
        plt.fill_between(x, curves[i], curves[i + 1], color=color, alpha=0.2, interpolate=True)
    plt.title(np.array(title))
    plt.legend()
    plt.show()

def show_cluster_2d(vectors, clusters, medoids, n_clusters):
    """
    Plots clustered data in 2D using the first two principal components.

    Parameters:
    vectors (array-like): Data points to plot.
    clusters (array-like): Cluster labels for each data point.
    medoids (array-like): Indices of medoid points.
    n_clusters (int): Number of clusters.
    """
    clusters[medoids] = n_clusters
    print("vectors", np.array(vectors).shape)
    PC1 = vectors[:, 0]
    PC2 = vectors[:, 1]
    print("PC1", np.array(PC1).shape)
    print("PC2", np.array(PC2).shape)
    plt.figure(figsize=(8, 6))
    plt.scatter(PC1, PC2, c=clusters, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D Scatter Plot of Principal Components (PCA)')
    plt.grid(True)
    plt.show()

def show_2d(vectors):
    """
    Plots data in 2D using the first two principal components.

    Parameters:
    vectors (array-like): Data points to plot.
    """
    PC1 = vectors[:, 0]
    PC2 = vectors[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(PC1, PC2, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D Scatter Plot of Principal Components (PCA)')
    plt.grid(True)
    plt.show()

def show_cluster_3d(vectors, clusters, medoids, n_clusters):
    """
    Plots clustered data in 3D.

    Parameters:
    vectors (array-like): Data points to plot.
    clusters (array-like): Cluster labels for each data point.
    medoids (array-like): Indices of medoid points.
    n_clusters (int): Number of clusters.
    """
    print(np.unique(clusters))
    cluster_colors = ['blue', 'red', 'green', 'orange', 'purple']
    colors = [cluster_colors[int(label)] for label in (clusters - 1).astype(int)]

    trace = go.Scatter3d(
        x=vectors[:, 0],
        y=vectors[:, 1],
        z=vectors[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.8,
        )
    )
    layout = go.Layout(
        title='Cluster Colors: Blue = Speed 1; Red = Speed 2; Green = Speed 3; Orange = Spontaneous speed; Purple = High speed',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def show_3d(X, labels):
    """
    Plots data in 3D with labels.

    Parameters:
    X (array-like): Data points to plot.
    labels (array-like): Labels for each data point.
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    label_color_map = {label: color for label, color in zip(labels, colors)}

    trace = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=label_color_map,
            opacity=0.8,
            colorbar=dict(title='Label')
        )
    )
    layout = go.Layout(
        title='Interactive 3D Scatter Plot with Labels',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
