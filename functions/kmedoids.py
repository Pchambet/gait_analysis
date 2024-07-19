import numpy as np

def kmedoidsplusplus(dist_matrix, n_clusters, random_state=42, n_local_trials=10):
    """
    Initializes the medoids using the k-medoids++ algorithm.

    Parameters:
    - dist_matrix (ndarray): Distance matrix where dist_matrix[i, j] is the distance between points i and j.
    - n_clusters (int): Number of clusters.
    - random_state (int): Random seed for reproducibility.
    - n_local_trials (int): Number of local trials for finding the best medoid.

    Returns:
    - indices (ndarray): Array of indices of the initial medoids.
    """
    np.random.seed(random_state)
    n_samples = dist_matrix.shape[0]
    
    if n_local_trials is None: 
        n_local_trials = 2 + int(np.log(n_clusters))
        
    center_id = np.argmin(dist_matrix.mean(axis=1))
    indices = np.full(n_clusters, -1, dtype=int)
    indices[0] = center_id 

    closest_dist = dist_matrix[indices[0], np.newaxis, :]
    current_pot = closest_dist.sum()

    for c in range(1, n_clusters):
        rand_vals = np.random.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_dist), rand_vals)
        np.clip(candidate_ids, None, closest_dist.size - 1, out=candidate_ids)

        distance_to_candidates = dist_matrix[candidate_ids, :]
        np.minimum(closest_dist, distance_to_candidates, out=distance_to_candidates)
        candidate_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidate_pot)
        current_pot = candidate_pot[best_candidate]
        closest_dist = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        
        indices[c] = best_candidate
    return indices 

class KMedoids: 
    """
    K-Medoids clustering object.

    Parameters:
    - K (int): Number of medoids. Default is 3.
    """
    
    def __init__(self, K=3):
        self.K = K
        self.loss = None
        self.labels = None
        self.medoids = None
        self.n_iter = None
        
    def __repr__(self):
        return f"KMedoids(K={self.K}):\n- loss={self.loss},\n- labels={self.labels},\n- medoids={self.medoids},\n- n_iter={self.n_iter}"

    def train(self, distance_matrix, medoids=None, init='random', 
              random_state=None, epsilon=1e-10, max_iter=30, verbose=False):
        """
        Train the K-Medoids model to find the optimal medoids.

        Parameters:
        - distance_matrix (ndarray): Distance matrix where dist_matrix[i, j] is the distance between points i and j.
        - medoids (list of int): List of indices of medoids to be used as initial medoids. Default is None.
        - init (str): Type of initialization ('random' or 'kmedoids++'). Default is 'random'.
        - random_state (int): Random seed for reproducibility. Default is None.
        - epsilon (float): Small value to determine the stopping criterion based on relative loss change. Default is 1e-10.
        - max_iter (int): Maximum number of iterations for the training. Default is 30.
        - verbose (bool): If True, prints the medoids at each step. Default is False.

        Returns:
        - medoids_indices_per_step (dict): Dictionary containing medoids indices at each step.
        - labels_per_step (dict): Dictionary containing labels at each step.
        - loss_per_step (dict): Dictionary containing loss at each step.
        """
        # Check if distance matrix is given as a numpy array
        if not isinstance(distance_matrix, np.ndarray):
            distance_matrix = np.array(distance_matrix)

        # Check if the number of medoids is an integer
        if not isinstance(self.K, int):
            raise ValueError("Wrong format for K. Please use an integer.")

        # Two kinds of initialization are possible: random or kmedoids++
        if init not in ['random', 'kmedoids++']:
            raise ValueError("The initialization is either random or kmedoids++.")

        if random_state is not None:
            np.random.seed(random_state)

        N = distance_matrix.shape[0]
        index = np.arange(N)

        medoids_indices_per_step = {}
        labels_per_step = {}
        loss_per_step = {}

        # Initialization
        if (medoids is None) and (init == 'random'):
            medoids = np.random.choice(N, self.K, replace=False)
        if (medoids is None) and (init == 'kmedoids++'):
            medoids = kmedoidsplusplus(distance_matrix, n_clusters=self.K, random_state=random_state, n_local_trials=distance_matrix.shape[0])
        else: 
            medoids = medoids
        
        # Assign labels based on the nearest medoid
        labels = np.array([np.argsort(distance_matrix[i, medoids])[0] for i in range(N)])

        new_loss = 0
        for i in range(self.K):
            new_loss += np.sum(distance_matrix[index[labels == i], medoids[i]])

        medoids_indices_per_step['step0'] = medoids
        labels_per_step['step0'] = labels
        loss_per_step['step0'] = new_loss

        loss = 1e20
        
        # Iteration
        step = 1
        if verbose:
            print(f"Initial step: Medoids' index {medoids}")

        while ((loss - new_loss) / loss > epsilon) and (step < max_iter):
            loss = new_loss
            new_loss = 0 
            labels = labels_per_step[f'step{step-1}']

            new_medoids = np.array([index[labels == i][distance_matrix[np.ix_(labels == i, labels == i)].sum(axis=1).argmin()] for i in range(self.K)])
            new_labels = np.array([np.argsort(distance_matrix[i, new_medoids])[0] for i in index])

            for i in range(self.K):
                new_loss += np.sum(distance_matrix[index[new_labels == i], new_medoids[i]])

            medoids_indices_per_step[f'step{step}'] = new_medoids
            labels_per_step[f'step{step}'] = new_labels
            loss_per_step[f'step{step}'] = new_loss
            if verbose:
                print(f"Step{step}: Medoids' index {new_medoids}")
            step += 1
        
        # Saving main results in the KMedoids object
        self.loss = new_loss
        self.labels = new_labels
        self.medoids = new_medoids
        self.n_iter = step
        
        return medoids_indices_per_step, labels_per_step, loss_per_step    
