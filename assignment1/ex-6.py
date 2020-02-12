import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)
POP_SIZE = 10

def plot_route(dataset, route):
    plt.scatter(dataset[:, 0], dataset[:, 1], s=5)
    plt.title("Map of cities with route")

    # route is list of cities, subtract 1 because they are not indices
    prev_coords = dataset[route[0]-1]
    colors = np.linspace(0, 1, len(dataset), True)
    for i in range(len(dataset)):
        coords = dataset[route[i]-1]
        x1, y1 = prev_coords
        x2, y2 = coords
        plt.plot([x1, x2], [y1, y2], 'k-', c=(colors[i], colors[len(dataset)-1-i], 0))
        prev_coords = coords
    plt.show()

def eucl_dist(coordinates, city1, city2):
    x1, y1 = coordinates[city1-1]
    x2, y2 = coordinates[city2-1]
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def total_eucl_dist(tour, coordinates):
    return sum([eucl_dist(coordinates, tour[i-1], tour[i]) for i in range(1,len(tour))])

def init_population(population_size, n_cities):
    return np.array([np.random.permutation(np.arange(1,n_cities+1)) for _ in range(population_size)])

def fitness(population, coordinates):
    return np.array([total_eucl_dist(tour, coordinates) for tour in population])

def select_parents(population, fitness):
    parents = []
    for _ in range(len(population)):
        # Binary tournament selection
        c1, c2 = np.random.randint(len(population), size=2)
        fittest = population[c1] if fitness[c1] < fitness[c2] else population[c2]
        parents.append(fittest)
    return np.array(parents)

def crossover(parents):
    n_cities = len(parents[0])
    children = []

    for _ in range(len(parents)):
        parents_crossover = parents[np.random.randint(len(parents), size=2)]
        # Randomly determine which parent is cur_parent and which parent is cur_parent
        if np.random.random() < 1:
            parents_crossover = np.flipud(parents_crossover)
            
        cur_parent = parents_crossover[0]
        other_parent = parents_crossover[1]
        child = np.zeros(n_cities, dtype=int)
        
        cut_points = np.random.randint(0,n_cities+1,2)
        if cut_points[0] > cut_points[1]:
            cut_points = np.flip(cut_points)
        cut_p1, cut_p2 = cut_points
        child[cut_p1:cut_p2] = cur_parent[cut_p1:cut_p2]
        # Take set difference such that there are no duplicate values, set assume_uniqe=True to prevent sorting
        remaining_cities = np.setdiff1d(np.roll(other_parent, -cut_p2), child, assume_unique=True)
        child[np.where(child==0)] = remaining_cities
        children.append(child)
    
    return np.array(children)

def mutation(children, mutation_probability=0.5):
    for child in children:
        if np.random.random() < mutation_probability:
            p1, p2 = np.random.choice(len(child), size=2, replace=False)
            v1 = child[p1]
            child[p1] = child[p2]
            child[p2] = v1
    return children

def evolve(dataset, iterations=10000, population_size=10):
    population = init_population(population_size, len(dataset))
    print(population, fitness(population, dataset))
    for i in range(iterations):
        fitness_vals = fitness(population, dataset)
        parents = select_parents(population, fitness_vals)
        children = crossover(parents)
        population = mutation(children)
        if i % 1000 == 0:
            print(i)
    return population

def main():
    coordinates_file = open("data/burma14.tsp.txt", "r")
    coordinates = np.array([line.split(" ")[1:] for line in coordinates_file.readlines()], dtype=int)
    # opt_route = np.array(open("data/bayg29.opt.tour.txt", "r").readlines(), dtype=int)

    final_population = evolve(coordinates, population_size=100, iterations=10000)
    final_fitness = fitness(final_population, coordinates)
    fittest = final_population[np.argmin(final_fitness, axis=0)]
    print(fittest, total_eucl_dist(fittest, coordinates))
    # print(opt_route, total_eucl_dist(opt_route, coordinates))

    plot_route(coordinates, fittest)
    # plot_route(coordinates, opt_route)

if __name__ == "__main__":
    main()
