import pandas as pd
import scipy.io
import numpy as np
import functions.methods as meth

def matlabToPython(vitesse = "Norm_V1.mat"):
    """
    Converts MATLAB data to Python format.

    Parameters:
    vitesse (str): Filename of the MATLAB file to be converted.

    Returns:
    np.array: Transformed data array.
    """
    # Define the directory of the MATLAB file
    directory = "/Users/pierrechambet/Desktop/gait_analysis/fmoissenet-PatternGenerator-2315600/Data/" + vitesse
    matrix_mat = scipy.io.loadmat(directory)

    # Extract normatives and kinematics data from the MATLAB file
    normatives = matrix_mat.get('Normatives')
    kinematics = normatives['Kinematics'][0, 0]
    fe3 = kinematics['FE3']
    data_fe3 = fe3[0, 0]['data'][0, 0]

    # Convert data to a Pandas DataFrame and then to a numpy array
    data = pd.DataFrame(data_fe3, index=range(np.array(data_fe3).shape[0]), columns=range(np.array(data_fe3).shape[1]))
    return np.array(-data.T)[:, :]

def matlabToPython_V2(X_medoids, vitesse = "Norm_V1.mat"):
    """
    Converts MATLAB data to Python format and performs Dynamic Time Warping (DTW) with reference.

    Parameters:
    X_medoids (np.array): Medoid data for DTW reference.
    vitesse (str): Filename of the MATLAB file to be converted.

    Returns:
    tuple: Identifiers, stance phase data, and transformed data array.
    """
    # Define the directory of the MATLAB file
    directory = "/Users/pierrechambet/Desktop/gait_analysis/fmoissenet-PatternGenerator-2315600/Data/" + vitesse
    matrix_mat = scipy.io.loadmat(directory)

    # Extract normatives and kinematics data from the MATLAB file
    normatives = matrix_mat.get('Normatives')

    # Data of knee
    kinematics = normatives['Kinematics'][0, 0]
    fe3 = kinematics['FE3']
    data_fe3 = fe3[0, 0]['data'][0, 0]
    data = pd.DataFrame(data_fe3, index=range(np.array(data_fe3).shape[0]), columns=range(np.array(data_fe3).shape[1]))
    data = np.array(-data.T)

    # Extract stance phase data
    gaitParameters = normatives['Gaitparameters'][0, 0]
    stance_phase = gaitParameters['stance_phase']
    stance_phase_data = stance_phase[0, 0]['data'][0, 0].T
    stance_phase_data = np.round(stance_phase_data)

    # Extract identifiers
    id = gaitParameters[0, 0]['sujets'][0, :]
    id_data = np.array([[item[0]] for item in id])

    # Perform DTW with reference
    data = meth.dtw_with_ref(X_medoids, vitesse)
    return id_data, stance_phase_data, data

def extract_identifiant(vitesse = "Norm_V1.mat"):
    """
    Extracts identifiers from the MATLAB data.

    Parameters:
    vitesse (str): Filename of the MATLAB file to be extracted.

    Returns:
    np.array: Array of identifiers and cycle numbers.
    """
    # Define the directory of the MATLAB file
    directory = "/Users/pierrechambet/Desktop/gait_analysis/fmoissenet-PatternGenerator-2315600/Data/" + vitesse
    matrix_mat = scipy.io.loadmat(directory)
    normatives = matrix_mat.get('Normatives')
    gaitParameters = normatives['Gaitparameters'][0, 0]

    # Extract identifiers
    id = gaitParameters[0, 0]['sujets'][0, :]
    id_data = np.array([[item[0]] for item in id])

    # Calculate cycle numbers
    numero_cycle = np.zeros(len(id_data))
    for i in range(1, len(id_data)):
        if id_data[i] == id_data[i-1]:
            numero_cycle[i] = numero_cycle[i - 1] + 1
        else:
            numero_cycle[i] = 0

    numero_cycle = numero_cycle.astype(int)
    data = np.column_stack((id_data, numero_cycle))
    return data

def phases(name = "Norm_V1.mat", interval = [0, 101]):
    """
    Extracts phases from the MATLAB data.

    Parameters:
    name (str): Filename of the MATLAB file to be extracted.
    interval (list): Interval range for data extraction.

    Returns:
    np.array: Extracted data within the specified interval.
    """
    X = matlabToPython(name)
    return X[:, interval[0]:interval[1]]
