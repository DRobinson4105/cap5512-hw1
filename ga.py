import random
from deap import base, creator, tools

random.seed(56)

POPULATION_SIZE = 100
INITIAL_POPULATION = [random.randint, 0, 1]
GENOME_LENGTH = 200
CROSSOVER_TYPE = [tools.cxOnePoint]
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
SELECTION_METHOD = [tools.selRoulette]
MAX_NUMBER_OF_GENERATIONS = 100

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", *INITIAL_POPULATION)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, GENOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", *CROSSOVER_TYPE)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE)
toolbox.register("select", *SELECTION_METHOD)

def main():
    pop = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0

    while max(fits) < GENOME_LENGTH and g < MAX_NUMBER_OF_GENERATIONS:
        g += 1
        print(f"Generation {g}")

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x ** 2 for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print(f"Minimum: {min(fits)}\nMaximum: {max(fits)}\nAverage: {mean}\nStandard Deviation: {std}\n")

if __name__ == "__main__":
    main()