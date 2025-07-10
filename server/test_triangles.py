import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# Load the CSV file
file_path = "./mcr/capture/data/all_points.csv"
df = pd.read_csv(file_path, header=None)
points = df.iloc[:, :3].values

# Run DBSCAN clustering
clustering = DBSCAN(eps=0.01, min_samples=5).fit(points)
labels = clustering.labels_

# Compute cluster centroids
centroids = []
for label in np.unique(labels):
    if label == -1:
        continue
    cluster_pts = points[labels == label]
    if len(cluster_pts) >= 3:
        centroid = np.mean(cluster_pts, axis=0)
        centroids.append(centroid)

centroids = np.array(centroids)

# Look for 3:4:5 triangles among centroids
triangle_points = []
target_ratios = sorted([3 / 4, 4 / 5, 5 / 3])
tolerance = 0.25

for A, B, C in combinations(centroids, 3):
    AB = np.linalg.norm(A - B)
    BC = np.linalg.norm(B - C)
    CA = np.linalg.norm(C - A)
    if min(AB, BC, CA) < 1e-6:
        continue
    ratios = sorted([AB / BC, BC / CA, CA / AB])
    if all(abs(r - t) / t < tolerance for r, t in zip(ratios, target_ratios)):
        triangle_points.append((A, B, C))

# Plot combined data and matching triangles
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="tab20", s=3, alpha=0.5
)

# Draw triangle edges between matched centroids
for A, B, C in triangle_points:
    tri = np.array([A, B, C, A])
    AB = np.linalg.norm(A - B)
    BC = np.linalg.norm(B - C)
    CA = np.linalg.norm(C - A)
    tri_ratios = sorted([AB / BC, BC / CA, CA / AB])
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color="black", linewidth=2)
    print(f"Triangle found: Ratios: {[round(r, 3) for r in tri_ratios]}")

print(f"Expected Ratios: {[round(r, 3) for r in target_ratios]}")

ax.set_title(
    f"3:4:5 Triangles Between Cluster Centroids ({len(triangle_points)} found)"
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()
