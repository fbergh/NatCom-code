import numpy as np
import matplotlib.pyplot as plt
import graphviz

from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

# Add the required functions and
# Add exponential function
def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)
exp = make_function(function=_protected_exponent, name='expo', arity=1)

# Add fitness function
def _fitness(y, y_pred, sample_weight):
    return np.sum(np.abs(y-y_pred))
fit = make_fitness(function=_fitness, greater_is_better=False, wrap=False)

# Returns the given dataset
def get_data():
    x = np.linspace(-1, 1, 21).reshape(-1,1)
    y = np.array([0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.0909, 0.0, 0.1111, 0.2496, 0.4251, 0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4.0000] )
    return x, y

# Define the parameters
population_size = 1000
function_set = ['add', 'sub', 'mul', 'div', 'log', 'cos', exp]
n_gens = 50
p_cross = 0.7
p_mut = 0

def run_experiment(seed, i):
    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=n_gens, stopping_criteria=0.01,
                               p_crossover=p_cross, p_subtree_mutation=p_mut,
                               p_hoist_mutation=p_mut, p_point_mutation=p_mut,
                               function_set = function_set,
                               max_samples=0.9, verbose=1,
                               metric=fit, random_state=seed)
    # Fit to the data
    est_gp.fit(x, y)

    # Get symbolic representation of the program
    print(est_gp._program)
    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('sym_{}.png'.format(seed))

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.xlabel('Generations', fontsize=24)
    plt.ylabel('Best fitness', fontsize=24)
    plt.plot(est_gp.run_details_['best_fitness'], linewidth=3.0)
    plt.grid()

    plt.subplot(1,2,2)
    plt.xlabel('Generations', fontsize=24)
    plt.ylabel('Best size', fontsize=24)
    plt.plot(est_gp.run_details_['best_length'], linewidth=3.0, color='red')
    plt.grid()

    plt.suptitle('Run {}'.format(i), fontsize=24)
    #plt.tight_layout()
    plt.savefig('plot_{}.eps'.format(seed))
    return est_gp.run_details_

if __name__ == '__main__':
    seeds = list(range(1335, 1345))
    x, y = get_data()
    run_details = []
    for i, seed in enumerate(seeds):
        print('Experiment {}: Seed {}'.format(i, seed))
        run_experiment(seed, i)
        print('\n\n')
