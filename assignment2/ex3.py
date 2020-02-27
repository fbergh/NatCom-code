import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datasets
from algorithms import kmeans, pso, quantization_error

print('Arificial Dataset')
x, y = datasets.artificial()
kmeans_errors = []
print(' --- KMeans')
for i in range(10):
    centroids, members = kmeans(x, nc=2, n_iter=30)
    kmeans_errors.append(quantization_error(x, centroids, members))

pso_errors = {5: [], 10: [], 20: []}
for p in pso_errors:
    print(' --- PSO {}'.format(p))
    for i in range(10):
        centroids, members = pso(x, nc=2, n_particles=p, n_iter=100)
        pso_errors[p].append(quantization_error(x, centroids, members))
df = pd.DataFrame({
    'KMeans': kmeans_errors,
    '5 Particles': pso_errors[5],
    '10 Particles': pso_errors[10],
    '20 Particles': pso_errors[20],
})
plt.figure(figsize=(8,5))
sns.barplot(data=df)
plt.savefig('ex3art.pdf')

print('IRIS Dataset')
x, y = datasets.iris()
kmeans_errors = []
print(' --- KMeans')
for i in range(10):
    centroids, members = kmeans(x, nc=3, n_iter=30)
    kmeans_errors.append(quantization_error(x, centroids, members))

pso_errors = {5: [], 10: [], 20: []}
for p in pso_errors:
    print(' --- PSO {}'.format(p))
    for i in range(10):
        centroids, members = pso(x, nc=3, n_particles=p, n_iter=100)
        pso_errors[p].append(quantization_error(x, centroids, members))
df = pd.DataFrame({
    'KMeans': kmeans_errors,
    '5 Particles': pso_errors[5],
    '10 Particles': pso_errors[10],
    '20 Particles': pso_errors[20],
})
plt.figure(figsize=(8,5))
sns.barplot(data=df)
plt.savefig('ex3iris.pdf')
