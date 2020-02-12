import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
POP_SIZE = 10

def plot_opt_route(dataset, opt_route):
    plt.scatter(dataset[:, 0], dataset[:, 1], s=5)
    plt.title("Map of cities with optimal route")

    # opt_route is list of cities, subtract 1 because they are not indices
    prev_coords = dataset[opt_route[0]-1]
    colors = np.linspace(0, 1, len(dataset), True)
    for i in range(len(dataset)):
        coords = dataset[opt_route[i]-1]
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
    return [np.random.permutation(n_cities) for _ in range(population_size)]

def fitness(population, coordinates):
    return [total_eucl_dist(tour, coordinates) for tour in population]

def select_parents(population, fitness):
    parents = []
    for _ in range(len(population)):
        # Binary tournament selection
        c1, c2 = np.random.randint(len(population), size=2)
        fittest = population[c1] if fitness[c1] > fitness[c2] else population[c2]
        parents.append(fittest)
    return np.array(parents)

def crossover(parents):
    n_cities = len(parents[0])
    children = []

    for _ in range(len(parents)):
        parents_crossover = parents[np.random.randint(len(parents), size=2)]
        # Randomly determine which parent is cur_parent and which parent is cur_parent
        if np.random.random() < 0.5:
            parents_crossover = np.flip(parents_crossover)

        cur_parent = parents_crossover[0]
        other_parent = parents_crossover[1]
        child = np.zeros(n_cities)

        # Generate cut_p1 under assumption cut_p1 < n_cities
        cut_p1 = np.random.randint(n_cities)
        # Generate cut_p2 under assumption cut_p2 >= cut_p1
        cut_p2 = np.random.randint(cut_p1, n_cities+1)
        child[cut_p1:cut_p2] = cur_parent[cut_p1:cut_p2]

        child[:cut_p1] = other_parent[:cut_p1]
        # To avoid out of bounds exception
        if cut_p2 != n_cities-1:
            child[cut_p2:] = other_parent[cut_p2:]
        
        children.append(child)
    return np.array(children)

def mutation(children, mutation_probability=0.05):
    for child in children:
        if np.random.random() < mutation_probability:
            p1, p2 = np.random.choice(len(child), size=2, replace=False)
            v1 = child[p1]
            child[p1] = child[p2]
            child[p2] = v1
    return children


def main():
    dataset_file = open("data/bayg29.tsp.txt", "r")
    dataset = np.array([line.split(" ")[1:] for line in dataset_file.readlines()], dtype=int)
    opt_route = np.array(open("data/bayg29.opt.tour.txt", "r").readlines(), dtype=int)

    population = init_population(POP_SIZE, len(dataset))
    population = np.array(population)
    fitness = fitness(population, dataset)
    parents = select_parents(population, fitness)

    plot_opt_route(dataset, opt_route)

if __name__ == "main":
    main()
