# Clinical Gait Analysis

## Overview

This project focuses on the analysis of gait data. The project includes various methods for extracting, clustering, and analyzing gait data, with potential applications in biometrics, health diagnostics, and motion analysis.

The initial project is organized into several Python, MATLAB and Javascript files, each responsible for different aspects of the data processing pipeline.

On this project, only the folder "Data" is be used. In fact, the goal of this project is to analyse the MATLAB data to theorize and classifiy the gait.

The data are composed of five MATLAB files that represent cycles of walk. 52 healthy persons (no neuro-orthopedic troubles) have been asked to walk under five different speed constraints. Each data file correspond to 10 cycles of these 52 persons for a given speed constraint.

The goal of this project is to analyse the impact of these speed constraints over a person's gait.

To that aim, the folder "functions" have been created where all the functions and methods of extracting, clustering, and analyzing gait data are defined.

## Project Structure
```plaintext
.
├── Data
│   ├── Norm_V1.mat
│   ├── Norm_V2.mat
│   ├── Norm_V3.mat
│   ├── Norm_V4.mat
│   └── Norm_V5.mat
├── functions
│   ├── ACP.py
│   ├── clustering.py
│   ├── DTW.py
│   ├── extract_data.py
│   ├── kmedoids.py
│   ├── methods.py
│   ├── show.py
│   └── variance.py
├── Presentation
│   ├── final_presentation.pdf
├── main.py
└── README.md
```

## Quick explanation of our work

- **main.py**: The main entry point of the project. It orchestrates the execution of various functions and methods defined in other files.
- **ACP.py**: Implements Principal Component Analysis (PCA) for dimensionality reduction and feature extraction.
- **clustering.py**: Contains functions and methods related to clustering algorithms, including K-Medoids and hierarchical clustering.
- **DTW.py**: Implements Dynamic Time Warping (DTW), used to measure similarity between temporal sequences and visualize the DTW path and cost matrix.
- **extract_data.py**: Handles data extraction from raw sources, preprocessing, and preparing it for analysis.
- **kmedoids.py**: Contains the implementation of the K-Medoids clustering algorithm, including a k-medoids++ initialization method.
- **methods.py**: Includes various helper methods for data visualization, clustering, and analysis.
- **show.py**: Contains functions for visualizing data and results.
- **variance.py**: Implements methods for analyzing the variance in data.

## Functionality

### Principal Component Analysis (PCA)
Implemented in `ACP.py`, PCA is used for reducing the dimensionality of the data, making it easier to visualize and analyze. It includes functions like `acp_components` for computing principal components, `compute_threshold` for determining the number of components needed to reach a variance threshold, `acp_threshold` for computing components that meet the threshold, and `acp` for projecting data onto the new basis.

### Clustering
`clustering.py` contains clustering algorithms, including:

- **clustering_with_dtw**: Clustering using Dynamic Time Warping (DTW) distance and K-Medoids.
- **clustering_with_euclidean**: Clustering using Euclidean distance and K-Medoids.
- **hierarchical_clustering**: Hierarchical clustering using DTW distance.
- **k_means**: Clustering using K-Means.

### Dynamic Time Warping (DTW)
Implemented in `DTW.py`, DTW is used to measure similarity between time-series data which may vary in speed. It includes functions like `compute_dtw` for calculating the DTW distance matrix, `plot_dtw` for visualizing the DTW path, and `plot_dtw_mat` for visualizing the DTW cost matrix.

### Data Extraction
`extract_data.py` handles the extraction and preprocessing of raw data, making it suitable for analysis. It includes functions like `matlabToPython` for converting MATLAB data to Python format, and `extract_identifiant` for extracting identifiers from the data.

### Visualization
`show.py` includes functions to visualize the data and the results of the analysis. It provides various methods to plot curves, clustered data in 2D and 3D, and labeled data in 3D.

### Variance Analysis
`variance.py` provides methods to analyze the variance in the data, helping to understand the importance of different features. It includes functions like `select_individu` for selecting individual data, `variance` for computing the variance, and `truncate` for truncating data based on stance phase length.

### Methods
`methods.py` includes various helper functions for data visualization, clustering, and analysis. Functions include:

- **dendrogramme**: Generates a dendrogram for hierarchical clustering.
- **show_data**: Shows the curve for the given data.
- **view_data**: Displays the first curve of the given data.
- **medoids**: Computes and shows the medoids for clustering.
- **fill_curves**: Fills between the curves on a 2D graph.
- **contribution_composantes**: Computes the contribution of components.
- **distribution**: Prints the distribution of data points in clusters.
- **plot_3D**: Plots data in 3D.
- **plot_2D**: Plots data in 2D.
- **plot_cluster_3D**: Plots clustered data in 3D.
- **plot_cluster_2D**: Plots clustered data in 2D.
- **dtw_with_ref**: Computes the DTW distance with reference medoids.
- **courbe**: Shows a specific curve.
- **view_3D**: Shows a 3D view of the data.
- **bar**: Plots a bar graph for segments.
- **plot_bar**: Plots a single bar for the segment.
- **segmentation**: Segments the data based on identifiers.
- **variance**: Computes the variance for different speeds.

