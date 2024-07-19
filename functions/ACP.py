import numpy as np

def acp_components(data, n_components):
    """
    Perform Principal Component Analysis (PCA) on the dataset and return the specified number of principal components.

    Parameters:
    data (array-like): The input data to perform PCA on.
    n_components (int): The number of principal components to return.

    Returns:
    array-like: The principal components of the data.
    """
    data_centred = data - np.mean(data, axis=0)
    cov_matrix = (1/data_centred.shape[0]) * np.dot(data_centred.T, data_centred)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    principal_components = sorted_eigenvectors[:, :n_components]

    for i in range(n_components):
        rate = 100 * sorted_eigenvalues[i] / np.sum(eigenvalues)
        print(f"Explained variance by component {i+1}: {round(rate, 1)}%")

    return principal_components.T

def compute_threshold(eigenvalues, threshold):
    """
    Compute the number of components required to reach the specified threshold of explained variance.

    Parameters:
    eigenvalues (array-like): The eigenvalues of the covariance matrix.
    threshold (float): The threshold of explained variance (between 0 and 1).

    Returns:
    int: The number of components needed to reach the threshold.
    """
    num_components = 0
    cumulative_variance = 0

    for value in eigenvalues:
        cumulative_variance += value
        num_components += 1
        if cumulative_variance / np.sum(eigenvalues) >= threshold:
            break

    return num_components

def acp_threshold(X, threshold):
    """
    Perform PCA and return the transformed data and the number of components required to reach the specified threshold.

    Parameters:
    X (array-like): The input data to perform PCA on.
    threshold (float): The threshold of explained variance (between 0 and 1).

    Returns:
    tuple: Transformed data and the number of components.
    """
    X = np.array(X)
    X_centred = X - np.mean(X, axis=0)
    cov_matrix = (1 / X_centred.shape[0]) * np.dot(X_centred.T, X_centred)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    num_components = compute_threshold(sorted_eigenvalues, threshold)
    print(f"Number of components needed: {num_components}")
    principal_components = sorted_eigenvectors[:, :num_components]

    X_new_basis = np.dot(X_centred, principal_components)
    return X_new_basis, num_components

def acp(X, threshold):
    """
    Perform PCA and project the original data onto the principal components basis.

    Parameters:
    X (array-like): The input data to perform PCA on.
    threshold (float): The threshold of explained variance (between 0 and 1).

    Returns:
    tuple: Projected data and the number of components.
    """
    basis, num_components = acp_threshold(X, threshold)
    print(f"Original shape of data: {np.array(X).shape}")
    print(f"Shape after PCA: {np.array(basis).shape}")
    X_proj = np.dot(X.T, basis)

    return X_proj, num_components

# README Section
"""
Principal Component Analysis (PCA) Implementation
-------------------------------------------------

This script implements Principal Component Analysis (PCA) for dimensionality reduction and feature extraction. 
It includes functions to compute the principal components, determine the number of components required to reach 
a specified threshold of explained variance, and project the data onto the principal components basis.

Functions:
- acp_components(data, n_components): Returns the specified number of principal components.
- compute_threshold(eigenvalues, threshold): Computes the number of components needed to reach the threshold of explained variance.
- acp_threshold(X, threshold): Performs PCA and returns the transformed data and the number of components.
- acp(X, threshold): Projects the original data onto the principal components basis.

Usage:
Import this module and call the functions with your dataset to perform PCA. Adjust the number of components or the threshold 
of explained variance as needed.

Example:
import numpy as np
from acp import acp

# Example data
data = np.random.rand(100, 5)

# Perform PCA with 95% explained variance threshold
projected_data, num_components = acp(data, 0.95)

print(f"Projected data shape: {projected_data.shape}")
print(f"Number of components: {num_components}")
"""
