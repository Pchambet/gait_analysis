import numpy as np

def acp_components(data, n_components):
    """
    Computes the principal components of the data.

    Parameters:
    - data (array-like): The input data.
    - n_components (int): The number of principal components to compute.

    Returns:
    - principal_components (array-like): The computed principal components.
    """
    data_centred = data - np.mean(data, axis=0)
    cov_matrix = (1 / data_centred.shape[0]) * np.dot(data_centred.T, data_centred)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    principal_components = sorted_eigenvectors[:, :n_components]

    for i in range(n_components):
        rate = 100 * sorted_eigenvalues[i] / np.sum(eigenvalues)
        print(round(rate, 1))

    return principal_components.T

def compute_threshold(eigenvalues, threshold):
    """
    Computes the number of principal components needed to reach the given threshold.

    Parameters:
    - eigenvalues (array-like): The eigenvalues of the covariance matrix.
    - threshold (float): The variance threshold.

    Returns:
    - num_components (int): The number of components needed to reach the threshold.
    """
    num_components = 0
    j = 0
    rate = 0
    while rate < threshold:
        j += eigenvalues[num_components]
        rate = j / np.sum(eigenvalues)
        num_components += 1
    return num_components

def acp_threshold(X, threshold):
    """
    Computes the principal components of the data that meet the variance threshold.

    Parameters:
    - X (array-like): The input data.
    - threshold (float): The variance threshold.

    Returns:
    - X_new_basis (array-like): The data in the new basis.
    - num_components (int): The number of components needed to reach the threshold.
    """
    X = np.array(X)
    X_centred = X - np.mean(X, axis=0)
    cov_matrix = (1 / X_centred.shape[0]) * np.dot(X_centred.T, X_centred)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    num_components = compute_threshold(sorted_eigenvalues, threshold)
    print("Nombre de composantes", num_components)
    principal_components = sorted_eigenvectors[:, :num_components]

    X_new_basis = np.dot(X_centred, principal_components)
    return X_new_basis, num_components

def acp(X, threshold):
    """
    Performs PCA on the data and projects it onto the new basis.

    Parameters:
    - X (array-like): The input data.
    - threshold (float): The variance threshold.

    Returns:
    - X_proj (array-like): The projected data.
    - num_components (int): The number of components needed to reach the threshold.
    """
    basis, num_components = acp_threshold(X, threshold)
    print(np.array(X).shape)
    print(np.array(basis).shape)
    X_proj = np.dot(X.T, basis)

    return X_proj, num_components
