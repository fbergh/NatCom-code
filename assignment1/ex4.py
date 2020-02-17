from deap import base
from deap import creator
from deap import tools 

import random
import matplotlib.pyplot as plt

COUNTING_ONES = 100

POPULATION_SIZE = 200

P_CROSSOVER = 0.9
P_MUTATION = 1.0 / COUNTING_ONES

MAX_GENERATIONS = 1500


toolbox = base.Toolbox()
toolbox.register("randomOneOrZero", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights = (1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomOneOrZero, COUNTING_ONES)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def oneMaxFitness(individual):
    return sum(individual), 

toolbox.register("evaluate", oneMaxFitness)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("crossover", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / COUNTING_ONES)
 
population = toolbox.populationCreator(n=POPULATION_SIZE)
interations_number = 0

fitnessValues = list(map(toolbox.evaluate, population))
for individual, fitnessValue in zip(population, fitnessValues) :
    individual.fitness.values = fitnessValue

fitnessValues = [individual.fitness.values[0] for individual in population]

maxFitnessValues = []
meanFitnessValues = []

while max(fitnessValues) < COUNTING_ONES and interations_number < MAX_GENERATIONS:
   interations_number = interations_number + 1
   offspring = toolbox.select(population, len(population))
   offspring = list(map(toolbox.clone, offspring))

   for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        
   for mutant in offspring:
        if random.random() < P_MUTATION:
            toolbox.mutate(mutant)
            del mutant.fitness.values
        
   newIndividuals = [ind for ind in offspring if not ind.fitness.valid]
   newFitnessValues = list(map(toolbox.evaluate, newIndividuals))
   for individual, fitnessValue in zip(newIndividuals, newFitnessValues):
        individual.fitness.values = fitnessValue
   population[:] = offspring
    
   fitnessValues = [ind.fitness.values[0] for ind in population]
    
   maxFitness = max(fitnessValues)
   meanFitness = sum (fitnessValues) / len(population)
   maxFitnessValues.append(maxFitness)
   meanFitnessValues.append(meanFitness)
   print ("-Number of Generations {} : Max Fitness = {}".format(interations_number, maxFitness))
    
    
   best_index = fitnessValues.index(max(fitnessValues))
   print("Optimal  = ", *population[best_index], "\n")
    
   plt.plot(maxFitnessValues, color='blue')
   plt.xlabel('Iterations number')
   plt.ylabel('Highest Fitness')
   plt.title('MAX Best fitness over Generations')
   plt.show()



