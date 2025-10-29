# Clinical Gait Analysis: Time-Series Clustering

This project applies unsupervised machine learning techniques to analyze clinical gait data. The goal is to investigate the impact of different walking speed constraints on human gait patterns, specifically focusing on knee flexion as a time-series.

This was a M1-level academic project for the "Artificial intelligence for data science" (SIC7002) course.

---

## Project Objective

[cite_start]The dataset contains 10 knee-flexion gait cycles from 52 healthy participants [cite: 99, 101][cite_start], each walking under 5 different speed constraints (from very slow to high speed) [cite: 102-107].

[cite_start]**The core objective was to: Analyze the impact of these speed constraints on a person's gait, using clustering methods to segment and understand pattern changes.** [cite: 109]

[cite_start]*Caption: Example of a single gait cycle (left) vs. all cycles for a given speed (right).* [cite: 112, 148]

---

## Methodology

My approach was a 3-step unsupervised learning pipeline:

1.  [cite_start]**Time-Series Clustering (K-Medoids):** I first applied **K-Medoids clustering (k=3)** directly to the time-series data for each speed group[cite: 152]. [cite_start]I used **Dynamic Time Warping (DTW)** as the distance metric[cite: 152], which is ideal for comparing time-series that are shifted or vary in speed.

2.  **Feature Engineering (DTW Distances):** To visualize all cycles in one common space, I created a new 3D feature set. I first established a "reference gait" by finding the 3 medoids (gait patterns) of the "spontaneous speed" group. [cite_start]Then, I calculated the DTW distance from *every* cycle to each of these 3 reference medoids[cite: 193]. This transformed each time-series into a 3-dimensional vector `(dist_to_medoid_1, dist_to_medoid_2, dist_to_medoid_3)`.

3.  [cite_start]**Cluster Analysis (K-Means):** With the data now in a 3D space, I applied a standard **K-Means algorithm (k=5)** [cite: 156, 245] to identify the main clusters across *all* speed groups and analyze their distribution.

---

## Key Findings & Results

### Finding 1: Gait shape deforms significantly at slow speeds.
The K-Medoids analysis showed that gait patterns are not just scaled, but their *shape* changes. [cite_start]Notably, the slower the speed, the smaller the first "bump" in the knee flexion curve. [cite: 183-184]

*Caption: The 3 medoid (central) gait patterns found for each of the 5 speed groups.*

### Finding 2: Speed groups are clearly separable in the 3D feature space.
The 3D plot of DTW distances shows a clear separation between the different speed constraints. The slowest speeds (blue) and the reference/high speeds (red/orange) occupy distinct regions of the feature space.

[cite_start]*Caption: All gait cycles plotted in the 3D feature space, colored by their original speed constraint.* [cite: 203-219]

### Finding 3: Slow speeds increase gait variance and "individuality."
The K-Means clustering and variance analysis provided the key insight:

* [cite_start]**Cycles become more individual-specific:** As seen in the table below, the slowest speeds (Speed 1) are spread across all 5 clusters, while the reference speed (Ref) is highly concentrated in just one cluster[cite: 247, 252].
* [cite_start]**Variance increases:** The average variance within a single person's 10 cycles is highest at the slowest speed (26.88) and lowest at high speed (8.31)[cite: 277, 291].

*Caption: Distribution of speed groups across the 5 K-Means clusters (left) and average intra-person variance by speed (right).*

---

## Conclusion

This analysis successfully demonstrated that gait patterns are highly sensitive to speed constraints.

[cite_start]The main takeaway is that **slow-speed constraints make gait patterns more distinct and individual-specific**[cite: 300]. [cite_start]This suggests that in a clinical setting, asking a patient to walk slowly could be a valuable technique to **highlight and discriminate gait abnormalities** that might be hidden during a normal, "spontaneous" walk[cite: 301].

---

## Project Structure
