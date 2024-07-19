import pandas as pd
import scipy.io
import numpy as np
import os

def matlabToPython(vitesse="Norm_V1.mat"):
    """
    Convert MATLAB data to Python-friendly format.

    Parameters:
    vitesse (str): Filename of the MATLAB data file.

    Returns:
    numpy.ndarray: Transformed data.
    """
    directory = "your directory" + vitesse
    matrix_mat = scipy.io.loadmat(directory)
    normatives = matrix_mat.get('Normatives')

    # Data of knee
    kinematics = normatives['Kinematics'][0, 0]
    fe3 = kinematics['FE3']
    data_fe3 = fe3[0, 0]['data'][0, 0]
    data = pd.DataFrame(data_fe3, index=range(np.array(data_fe3).shape[0]), columns=range(np.array(data_fe3).shape[1]))
    data = -data.T
    print(f"Shape of knee data: {data.shape}")

    # Stance Phase
    gaitParameters = normatives['Gaitparameters'][0, 0]
    stance_phase = gaitParameters['stance_phase']
    stance_phase_data = stance_phase[0, 0]['data'][0, 0].T
    print(f"Shape of stance phase data: {stance_phase_data.shape}")

    # ID
    id = gaitParameters[0, 0]['sujets'][0, :]
    id_data = np.array([[item[0]] for item in id])
    print(f"Shape of ID data: {id_data.shape}")

    combined_array = np.array([id_data, stance_phase_data, data])
    print(f"Shape of combined array: {combined_array.shape}")
    return np.array(data)

def phases(name="Norm_V1.mat", interval=[0, 101]):
    """
    Extract phases from the data within a specified interval.

    Parameters:
    name (str): Filename of the MATLAB data file.
    interval (list): Interval to extract the phases.

    Returns:
    numpy.ndarray: Data within the specified interval.
    """
    X = matlabToPython(name)
    return X[:, interval[0]:interval[1]]

def access(vitesse, categorie, name):
    """
    Access specific category of data from the MATLAB file.

    Parameters:
    vitesse (str): Filename of the MATLAB data file.
    categorie (str): Category of data to access.
    name (str): Name of the data to access.

    Returns:
    numpy.ndarray: Accessed data.
    """
    directory = "your directory" + vitesse
    matrix_mat = scipy.io.loadmat(directory)
    normatives = matrix_mat.get('Normatives')
    category = normatives[categorie][0, 0]
    data = category[name][0, 0]['data'][0, 0]
    data = pd.DataFrame(data, index=range(np.array(data).shape[0]), columns=range(np.array(data).shape[1]))
    return np.array(-data.T)

def matlabToPython_V2(X_medoids, vitesse="Norm_V1.mat"):
    """
    Convert MATLAB data to Python format and align it with given medoids.

    Parameters:
    X_medoids (numpy.ndarray): Medoids to align the data with.
    vitesse (str): Filename of the MATLAB data file.

    Returns:
    tuple: ID data, stance phase data, and aligned data.
    """
    directory = "your directory" + vitesse
    matrix_mat = scipy.io.loadmat(directory)
    normatives = matrix_mat.get('Normatives')

    # Data of knee
    kinematics = normatives['Kinematics'][0, 0]
    fe3 = kinematics['FE3']
    data_fe3 = fe3[0, 0]['data'][0, 0]
    data = pd.DataFrame(data_fe3, index=range(np.array(data_fe3).shape[0]), columns=range(np.array(data_fe3).shape[1]))
    data = np.array(-data.T)

    # Stance Phase
    gaitParameters = normatives['Gaitparameters'][0, 0]
    stance_phase = gaitParameters['stance_phase']
    stance_phase_data = stance_phase[0, 0]['data'][0, 0].T
    stance_phase_data = np.round(stance_phase_data)

    # ID
    id = gaitParameters[0, 0]['sujets'][0, :]
    id_data = np.array([[item[0]] for item in id])

    # Align with medoids
    # Assuming meth.dtw_with_ref is a function from another module that needs to be imported
    # data = meth.dtw_with_ref(X_medoids, vitesse)

    return id_data, stance_phase_data, data

def extract_identifiant(vitesse="Norm_V1.mat"):
    """
    Extract subject identifiers and cycle numbers from the data.

    Parameters:
    vitesse (str): Filename of the MATLAB data file.

    Returns:
    numpy.ndarray: Array containing identifiers and cycle numbers.
    """
    directory = "your directory" + vitesse
    matrix_mat = scipy.io.loadmat(directory)
    normatives = matrix_mat.get('Normatives')
    gaitParameters = normatives['Gaitparameters'][0, 0]
    id = gaitParameters[0, 0]['sujets'][0, :]
    id_data = np.array([[item[0]] for item in id])

    numero_cycle = np.zeros(len(id_data))

    for i in range(1, len(id_data)):
        if id_data[i] == id_data[i-1]:
            numero_cycle[i] = numero_cycle[i - 1] + 1
        else:
            numero_cycle[i] = 0

    numero_cycle = numero_cycle.astype(int)
    data = np.column_stack((id_data, numero_cycle))
    return data
