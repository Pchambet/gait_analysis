import numpy as np
import matplotlib.pyplot as plt
import functions.extract_data as ed
import functions.show as show
import functions.clustering as clus
import functions.variance as var
from scipy.spatial import distance_matrix
import functions.ACP as acp

def dendrogramme(vitesse, interval=0):
    """
    Generates a dendrogram for hierarchical clustering.

    Parameters:
    vitesse (str): The speed category of the data.
    interval (list): Interval range for data extraction.
    """
    if interval != 0:
        X = ed.phases(vitesse, interval)
    else:
        X = ed.matlabToPython(vitesse=vitesse)

    clus.hierarchical_clustering(X)

def show_data(vitesse, interval=0):
    """
    Shows the curve for the given data.

    Parameters:
    vitesse (str): The speed category of the data.
    interval (list): Interval range for data extraction.
    """
    X = ed.matlabToPython(vitesse=vitesse)
    X = ed.phases(vitesse, interval)

    show.show_curve(X)

def view_data(vitesse):
    """
    Displays the first curve of the given data.

    Parameters:
    vitesse (str): The speed category of the data.
    """
    X = ed.matlabToPython(vitesse=vitesse)
    show.show_curve(X[0], title="")

def medoids(vitesse, n_components):
    """
    Computes and shows the medoids for clustering.

    Parameters:
    vitesse (str): The speed category of the data.
    n_components (int): Number of components for clustering.
    """
    X = ed.matlabToPython(vitesse=vitesse)

    medoids, labels = clus.clustering_with_dtw(X, n_components)
    print(medoids)
    show.show_curve(X[medoids['step0']])
    return X[medoids['step0']]

def fill_curves(vitesse, n_components):
    """
    Fills between the curves on a 2D graph.

    Parameters:
    vitesse (str): The speed category of the data.
    n_components (int): Number of components for clustering.
    """
    X = ed.matlabToPython(vitesse=vitesse)

    medoids, labels = clus.clustering_with_dtw(X, n_components)

    labels = [np.where(labels['step0'] == i)[0] for i in range(n_components)]
    nb_labels = [len(labels[i]) for i in range(n_components)]

    color = ['blue', 'red', 'green']
    for i in range(n_components):
        show.show_fill_curve(X[medoids['step0']], n_components, curves=X[labels[i]], nb_label=nb_labels, color=color[i])

def contribution_composantes(vitesse, threshold, interval=0):
    """
    Computes the contribution of components.

    Parameters:
    vitesse (str): The speed category of the data.
    threshold (float): The variance threshold for PCA.
    interval (list): Interval range for data extraction.
    """
    if interval != 0:
        X = ed.phases(vitesse, interval)
    else:
        X = ed.matlabToPython(vitesse=vitesse)

    X_new_basis, num_components = acp.acp_threshold(X, threshold)
    acp.acp_components(X, num_components)

def distribution(labels, n_components):
    """
    Prints the distribution of data points in clusters.

    Parameters:
    labels (dict): Dictionary of cluster labels.
    n_components (int): Number of components for clustering.
    """
    tab = []
    print("Distribution")
    for i in range(n_components):
        print("- n°", i, ":", np.where(labels['step0'] == i)[0].shape)
        tab.append(np.where(labels['step0'] == i)[0].shape)
    return tab

def plot_3D(vitesse):
    """
    Plots data in 3D.

    Parameters:
    vitesse (str): The speed category of the data.
    """
    X = ed.matlabToPython(vitesse)
    basis = acp.acp_components(X, 3)
    X = np.dot(X, basis.T)
    show.show_3d(X)

def plot_2D(vitesse, nb_cluster, interval=0):
    """
    Plots data in 2D.

    Parameters:
    vitesse (str): The speed category of the data.
    nb_cluster (int): Number of clusters for PCA.
    interval (list): Interval range for data extraction.
    """
    if interval != 0:
        X = ed.phases(vitesse, interval)
    else:
        X = ed.matlabToPython(vitesse=vitesse)

    basis = acp.acp_components(X, nb_cluster)
    X = np.dot(X, basis.T)
    show.show_2d(X)

def plot_cluster_3D(vitesse, nb_cluster, interval=0):
    """
    Plots clustered data in 3D.

    Parameters:
    vitesse (str): The speed category of the data.
    nb_cluster (int): Number of clusters.
    interval (list): Interval range for data extraction.
    """
    if interval != 0:
        X = ed.phases(vitesse, interval)
    else:
        X = ed.matlabToPython(vitesse=vitesse)

    medoids, labels = clus.clustering_with_dtw(X, nb_cluster)
    basis = acp.acp_components(X, 3)
    X = np.dot(X, basis.T)
    print(medoids['step0'])
    show.show_cluster_3d(X, labels['step0'], medoids['step0'], nb_cluster)

def plot_cluster_2D(vitesse, nb_cluster, interval=0):
    """
    Plots clustered data in 2D.

    Parameters:
    vitesse (str): The speed category of the data.
    nb_cluster (int): Number of clusters.
    interval (list): Interval range for data extraction.
    """
    if interval != 0:
        X = ed.phases(vitesse, interval)
    else:
        X = ed.matlabToPython(vitesse=vitesse)

    medoids, labels = clus.clustering_with_dtw(X, nb_cluster)
    basis = acp.acp_components(X, 2)
    X = np.dot(X, basis.T)
    print(medoids['step0'])
    show.show_cluster_2d(X, labels['step0'], medoids['step0'], nb_cluster)

def dtw_with_ref(X_medoids, vitesse):
    """
    Computes the DTW distance with reference medoids.

    Parameters:
    X_medoids (array-like): Reference medoid data.
    vitesse (str): The speed category of the data.
    """
    X = ed.matlabToPython(vitesse=vitesse)
    X_v = []
    for i in range(len(X)):
        X_v.append([
            distance_matrix(x=[X_medoids[0], X[i]], y=[X_medoids[0], X[i]], p=2)[1][0],
            distance_matrix(x=[X_medoids[1], X[i]], y=[X_medoids[1], X[i]], p=2)[1][0],
            distance_matrix(x=[X_medoids[2], X[i]], y=[X_medoids[2], X[i]], p=2)[1][0]
        ])
    return np.array(X_v)

def courbe(vitesse, n):
    """
    Shows a specific curve.

    Parameters:
    vitesse (str): The speed category of the data.
    n (int): Index of the curve to show.
    """
    X = ed.matlabToPython(vitesse=vitesse)
    print(X.shape)
    print(n)
    show.show_curve([X[n]])

def view_3D():
    """
    Shows 3D view of the data.
    """
    X_medoids = medoids(vitesse="Norm_V4.mat", n_components=3)

    V1_3_coord = dtw_with_ref(X_medoids, "Norm_V1.mat")
    V1_identifiant = ed.extract_identifiant()
    V2_3_coord = dtw_with_ref(X_medoids, "Norm_V2.mat")
    V3_3_coord = dtw_with_ref(X_medoids, "Norm_V3.mat")
    V4_3_coord = dtw_with_ref(X_medoids, "Norm_V4.mat")
    V5_3_coord = dtw_with_ref(X_medoids, "Norm_V5.mat")

    X = np.vstack((V1_3_coord, V2_3_coord, V3_3_coord, V4_3_coord, V5_3_coord))
    labels = np.concatenate([np.ones(len(V1_3_coord)), np.full(len(V2_3_coord), 2), np.full(len(V3_3_coord), 3), np.full(len(V4_3_coord), 4), np.full(len(V5_3_coord), 5)])

    n_clusters = 5
    n_vitesse = 5

    clusters = clus.k_means(X, n_clusters) + 1

    V1 = np.column_stack((V1_identifiant[:, 0], V1_identifiant[:, 1], clusters[:len(V1_3_coord)]))
    segments = segmentation(V1)
    bar(segments)
    for c in range(n_clusters):
        print("Cluster", c, end=": \t")
        for v in range(n_vitesse):
            print(np.where(np.logical_and(clusters[:] == c + 1, labels[:] == v + 1))[0].shape[0], end="\t")
        print("")

    show.show_cluster_3d(X, labels, 0, n_clusters)
    show.show_cluster_3d(X, clusters, 0, n_clusters)

def bar(segments):
    """
    Plots a bar graph for segments.

    Parameters:
    segments (list): List of segments to plot.
    """
    fig, ax = plt.subplots()

    for y, segment in enumerate(segments):
        indice_couleur = [sub_array[2] for sub_array in segment]
        plot_bar(indice_couleur, ax, y)

    ax.axis('off')
    plt.show()

def plot_bar(segment, ax, y):
    """
    Plots a single bar for the segment.

    Parameters:
    segment (list): Segment data.
    ax (matplotlib.axes.Axes): Matplotlib axes object.
    y (int): Y-axis position for the bar.
    """
    couleur_mapping = {
        '1': 'blue',
        '2': 'red',
        '3': 'green',
        '4': 'orange',
        '5': 'purple',
        '6': 'pink',
        '7': 'pink',
        '8': 'brown',
        '9': 'gray',
        '0': 'black'
    }

    for i in range(len(segment)):
        seg_couleurs = [couleur_mapping[chiffre] for chiffre in segment]
        ax.plot([i, i + 1], [y, y], color=seg_couleurs[i], linewidth=5)

from collections import defaultdict

def segmentation(data):
    """
    Segments the data based on identifiers.

    Parameters:
    data (array-like): Data to segment.

    Returns:
    list: Segmented data.
    """
    grouped_elements = defaultdict(list)
    for element in data:
        id_ = element[0]
        grouped_elements[id_].append(element)

    result = list(grouped_elements.values())
    return result

def variance():
    """
    Computes the variance for different speeds.
    """
    vitesse = ["Norm_V1.mat", "Norm_V2.mat", "Norm_V3.mat", "Norm_V4.mat", "Norm_V5.mat"]

    X_medoids = medoids(vitesse="Norm_V4.mat", n_components=3)

    V1_3_coord = dtw_with_ref(X_medoids, "Norm_V1.mat")
    V2_3_coord = dtw_with_ref(X_medoids, "Norm_V2.mat")
    V3_3_coord = dtw_with_ref(X_medoids, "Norm_V3.mat")
    V4_3_coord = dtw_with_ref(X_medoids, "Norm_V4.mat")
    V5_3_coord = dtw_with_ref(X_medoids, "Norm_V5.mat")
    print(V1_3_coord.shape)

    print("Entire cycle: ")
    for i in vitesse:
        print(i[5:7], end=": ")
        id_data, stance_phase_data, data = ed.matlabToPython_V2(X_medoids, vitesse=i)
        C3_coord = dtw_with_ref(X_medoids, i)
        C1_coord = np.mean(C3_coord, axis=1)
        C1_coord = C1_coord[:, np.newaxis]
        combined_array = np.concatenate((id_data, stance_phase_data, C1_coord), axis=1)

        data_individus, data_stances, id_unique = var.select_individu(combined_array)

        print(var.variance(data_individus))
