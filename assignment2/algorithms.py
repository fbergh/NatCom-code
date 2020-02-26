import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import datasets

def quantization_error(z, centroids, members):
    ''' Quantization error
    z : Data that are clustered with shape (n_samples, n_features)
    cetroids : Centroids for each cluster with shape (n_clusters, n_features)
    members : Index of cluster that each sample of z belongs to with shape (n_samples, 1)
    '''
    total = 0
    for i, c in enumerate(centroids):
        c_members = members[np.where(members == i)]
        total += np.mean(euclidean_distances(z, c[np.newaxis,:]))
    return total / len(centroids)

def find_closest(sample, centroids):
    ''' Returns the index of the centroid which is the closest to the sample '''
    dist = euclidean_distances(sample[np.newaxis,:], centroids)
    return np.argmin(dist)

def calculate_velocity(v, local_best, global_best, x):
    ''' Calculates the velocity given the current velocity and the global & local best
    '''
    w = 0.72
    c1 = c2 = 1.49
    r1, r2 = np.random.random(2)
    return w*v + c1*r1*(local_best - x) + c2*r2*(global_best - x)

def kmeans(z, nc=2, n_iter=100, return_errors=False):
    ''' Kmeans algorithm. Uses the notation defined by van der Merwe et al. '''

    no, nd = z.shape
    # Random initialization of centroids
    m = np.random.rand(nc, nd)
    cluster_members = np.zeros((no,), dtype=np.int16)

    errors = np.zeros(n_iter)

    for i in range(n_iter):
        # Find which cluster each sample belongs
        for p, zp in enumerate(z):
            closest = find_closest(zp, m)
            cluster_members[p] = closest

        # Update centroids
        for c in range(nc):
            ind = np.where(cluster_members == c)[0]
            if len(ind) > 0:
                m[c] = np.mean(z[ind])

        errors[i] = quantization_error(z, m, cluster_members)

    if return_errors:
        return m, cluster_members, errors
    else:
        return m, cluster_members

def pso(z, nc=2, n_particles=2, n_iter=100):
    ''' Particle swarm optimization algorithm. Uses the notation defined
        by van der Merwe et al. '''

    no, nd = z.shape
    x = np.random.uniform(low=-1, high=1, size=(n_particles, nc, nd))
    members = np.zeros((n_particles, no,))
    global_members = np.zeros((no,))
    local_best = np.zeros((n_particles, nc, nd))
    local_best_fit = np.full((n_particles,), np.inf)
    global_best = np.zeros((nc, nd))
    global_best_fit = np.inf
    v = np.zeros((n_particles, nc, nd))

    for t in range(n_iter):
        for i, xi in enumerate(x):
            for p, zp in enumerate(z):
                closest = find_closest(zp, xi)
                members[i, p] = closest

            # Update cluster centroids
            v[i] = calculate_velocity(v[i], local_best[i], global_best, xi)
            x[i] = xi + v[i]

            # Calculate fitness
            fit = quantization_error(z, xi, members[i])

            # Update local best
            if fit < local_best_fit[i]:
                local_best_fit[i] = fit
                local_best[i] = xi

                # Update global best
                if fit < global_best_fit:
                    global_best_fit = fit
                    global_best = xi
                    global_members = members[i]



    return global_best, global_members

if __name__ == "__main__":
    print('Artificial Dataset')
    x, y = datasets.artificial()
    centroids, members, errors = kmeans(x, nc=2, n_iter=80, return_errors=True)
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.xlabel('Iterations', fontsize=16)
    plt.xlabel('Quantization Error', fontsize=16)
    plt.savefig('artError.pdf')
    print(' --- KMeans', errors[-1])
    centroids, members = pso(x, nc=2, n_particles=10, n_iter=100)
    print(' --- PSO', quantization_error(x, centroids, members))

    print('Iris')
    x, y = datasets.iris()
    centroids, members, errors = kmeans(x, nc=3, n_iter=80, return_errors=True)
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.xlabel('Iterations', fontsize=16)
    plt.xlabel('Quantization Error', fontsize=16)
    plt.savefig('irisError.pdf')
    print(' --- KMeans', errors[-1])
    centroids, members = pso(x, nc=3, n_particles=10, n_iter=100)
    print(' --- PSO', quantization_error(x, centroids, members))
