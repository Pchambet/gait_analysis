import numpy as np

def select_individu(data):
    """
    Selects individual data and separates it into data and stance phases.

    Parameters:
    data (array-like): Input data containing identifiers, stance phases, and other data.

    Returns:
    tuple: (data_individus, data_stances, id_unique)
        - data_individus: List of arrays for each individual, each containing cycles of data.
        - data_stances: List of arrays for each individual, each containing stance phase values.
        - id_unique: Unique identifiers for each individual.
    """
    id_unique = np.unique(data[:, 0])
    identifiers = data[:, 0]

    data_individus = []
    data_stances = []

    for i, name in enumerate(id_unique):
        mask = (identifiers == name)
        individu = data[mask]

        data_string = individu[:, 2:]
        stance_string = individu[:, 1]

        data_as_floats = data_string.astype(float)
        stance_as_floats = stance_string.astype(float)

        data_individus.append(data_as_floats)
        data_stances.append(stance_as_floats)

    return data_individus, data_stances, id_unique

def variance(data):
    """
    Computes the variance for each individual and returns the mean variance.

    Parameters:
    data (list of arrays): List of arrays where each array contains cycles of data for an individual.

    Returns:
    float: The mean variance across all individuals, rounded to 2 decimal places.
    """
    list_variance_individus = []
    for index, cycles in enumerate(data):
        mean = sum(cycles) / len(cycles)
        somme_carres_ecarts = sum((x - mean) ** 2 for x in cycles)
        variance = somme_carres_ecarts / len(cycles)
        list_variance_individus.append(variance)

    variance_vitesse = np.mean(list_variance_individus)
    return round(variance_vitesse, 2)

def truncate(data, stance):
    """
    Truncates data based on the minimum stance phase length.

    Parameters:
    data (list of arrays): List of arrays where each array contains cycles of data for an individual.
    stance (list of arrays): List of arrays where each array contains stance phase values for an individual.

    Returns:
    list: Truncated data for each individual.
    """
    data_individus = []

    for i, individu in enumerate(data):
        min_stance = np.min(stance[i]).astype(int)
        individu_truncate = individu[:, :min_stance]

        data_individus.append(individu_truncate)

    return data_individus
