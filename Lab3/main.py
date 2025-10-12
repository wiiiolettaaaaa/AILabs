import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

centers = [[0, 4], [4, 7], [8, 3]]
data,_ = make_blobs(n_samples=800, centers=centers, random_state=42)

plt.scatter(data[:, 0], data[:, 1])
plt.title('Згенеровані дані')
plt.show()

center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, 3, 3, error=0.005, maxiter=100)
fuzzy_labels = np.argmax(u, axis=0)

for i in range(3):
    cluster_points = data[fuzzy_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

plt.scatter(center[:, 0], center[:, 1], marker='*', color='black')
plt.title('Кластери з центрами')
plt.show()

plt.plot(jm)
plt.xlabel('Кількість ітерацій')
plt.ylabel('Значення цільової функції')
plt.grid(True)
plt.show()