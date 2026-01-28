import random
from operator import itemgetter, attrgetter

from deap import base, creator, tools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

initial_config = {
    "POPULATION_SIZE": 100,
    "INITIAL_POPULATION": [random.randint, {"a": 0, "b": 1}],
    "GENOME_LENGTH": 200,
    "CROSSOVER_TYPE": [tools.cxOnePoint, {}],
    "CROSSOVER_RATE": 0.8,
    "MUTATION_RATE": 0.005,
    "SELECTION_METHOD": [tools.selRoulette, {}],
    "MAX_NUMBER_OF_GENERATIONS": 100,
    "RUNS": 50,
}

optimal_config = {
    "POPULATION_SIZE": 400,
    "INITIAL_POPULATION": [random.randint, {"a": 0, "b": 1}],
    "GENOME_LENGTH": 200,
    "CROSSOVER_TYPE": [tools.cxOnePoint, {}],
    "CROSSOVER_RATE": 1,
    "MUTATION_RATE": 0.005,
    "SELECTION_METHOD": [tools.selTournament, {"tournsize": 100}],
    "MAX_NUMBER_OF_GENERATIONS": 100,
    "RUNS": 50,
}

def run_once(config):
    (
        POPULATION_SIZE,
        INITIAL_POPULATION,
        GENOME_LENGTH,
        CROSSOVER_TYPE,
        CROSSOVER_RATE,
        MUTATION_RATE,
        SELECTION_METHOD,
        MAX_NUMBER_OF_GENERATIONS,
    ) = itemgetter(
        "POPULATION_SIZE",
        "INITIAL_POPULATION",
        "GENOME_LENGTH",
        "CROSSOVER_TYPE",
        "CROSSOVER_RATE",
        "MUTATION_RATE",
        "SELECTION_METHOD",
        "MAX_NUMBER_OF_GENERATIONS",
    )(config)

    creator.__dict__.pop("FitnessMax", None)
    creator.__dict__.pop("Individual", None)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", INITIAL_POPULATION[0], **INITIAL_POPULATION[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, GENOME_LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        return sum(individual),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", CROSSOVER_TYPE[0], **CROSSOVER_TYPE[1])
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE)
    toolbox.register("select", SELECTION_METHOD[0], **SELECTION_METHOD[1])

    pop = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    best = max(fits)

    g = 0

    best_fitness = [best]
    mean_fitness = [mean]
    first_optimum = 0 if best == GENOME_LENGTH else -1

    while g < MAX_NUMBER_OF_GENERATIONS:
        g += 1

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
        best = max(fits)

        best_fitness.append(best)
        mean_fitness.append(mean)

        if first_optimum == -1 and best == GENOME_LENGTH:
            first_optimum = g

    return {
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "first_optimum": first_optimum
    }

def evaluate(config, file_name, title, best_fitness_list, mean_fitness_list, horizontal=True):
    best_fitness_arr = np.array(best_fitness_list)
    mean_fitness_arr = np.array(mean_fitness_list)

    x = np.arange(config["MAX_NUMBER_OF_GENERATIONS"]+1)

    if horizontal:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax = np.ravel(ax)

    mean = np.mean(best_fitness_arr, axis=0)

    std = np.std(best_fitness_arr, axis=0, ddof=1)
    conf = 1.96 * std / np.sqrt(config["RUNS"])

    print(f"Best Fitness: {mean[-1]}\nStandard Deviation: {std[-1]}\nConfidence Interval: [{mean[-1] - conf[-1]}, {mean[-1] + conf[-1]}]")

    ax[0].plot(x, mean)
    ax[0].fill_between(x, mean - conf, mean + conf, alpha=0.3)

    ax[0].set_title("Best Fitness with 95% Confidence", fontsize=16)
    ax[0].set_xlabel("Generation", fontsize=14)
    ax[0].set_ylabel("Fitness Score", fontsize=14)
    ax[0].tick_params(axis="both", labelsize=14)

    mean = np.mean(mean_fitness_arr, axis=0)

    std = np.std(mean_fitness_arr, axis=0, ddof=1)
    conf = 1.96 * std / np.sqrt(config["RUNS"])

    ax[1].plot(x, mean)
    ax[1].fill_between(x, mean - conf, mean + conf, alpha=0.3)
    ax[1].set_title("Mean Fitness with 95% Confidence", fontsize=16)
    ax[1].set_xlabel("Generation", fontsize=14)
    ax[1].set_ylabel("Fitness Score", fontsize=14)
    ax[1].tick_params(axis="both", labelsize=14)

    fig.suptitle(title, fontsize=18)

    plt.tight_layout()
    plt.savefig(f"figures/{file_name}.png", dpi=300)

def run_many(config, file_name, title, horizontal=True, print_best_fitness=False):
    print(f"Running {file_name}")
    best_fitness_list = []
    mean_fitness_list = []
    first_optimum_list = []

    for i, _ in enumerate(tqdm(range(config["RUNS"]))):
        random.seed(i)
        run = run_once(config)
        best_fitness_list.append(run["best_fitness"])
        mean_fitness_list.append(run["mean_fitness"])
        if run["first_optimum"] != -1:
            first_optimum_list.append(run["first_optimum"])

    if print_best_fitness:
        print(f"Absolute best fitness is {max(max(x) for x in best_fitness_list)}")

    if len(first_optimum_list) > 0:
        mean_first_optimum = np.mean(np.array(first_optimum_list))
        print(f"Average generation of first optimum is {mean_first_optimum}")
    else:
        print("No run had an optimum individual")
    
    evaluate(config, file_name, title, best_fitness_list, mean_fitness_list, horizontal)

def selRank(individuals, k, fit_attr="fitness"):
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_ranks = len(individuals) * (len(individuals) + 1) // 2
    chosen = []

    for _ in range(k):
        u = random.random() * sum_ranks
        sum_ = 0
        for ind, rank in zip(s_inds, range(len(individuals), 0, -1)):
            sum_ += rank
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen
    
def part1():
    run_many(initial_config, "part1", "Initial Experiment", True)

def part2():
    populations = [50, 200, 400]
    parts = ['a', 'b', 'c']

    for pop, part in zip(populations, parts):
        config = initial_config.copy()
        config["POPULATION_SIZE"] = pop

        run_many(config, f"part2{part}", f"Population Size = {pop}", False)
        
def part3():
    mutation_rates = [0.0001, 0.001, 0.01]
    parts = ['a', 'b', 'c']

    for mut, part in zip(mutation_rates, parts):
        config = initial_config.copy()
        config["MUTATION_RATE"] = mut

        run_many(config, f"part3{part}", f"Mutation Rate = {mut}", False)

def part4():
    crossover_rates = [0.4, 0.6, 1.0]
    parts = ['a', 'b', 'c']

    for cor, part in zip(crossover_rates, parts):
        config = initial_config.copy()
        config["CROSSOVER_RATE"] =  cor

        run_many(config, f"part4{part}", f"Crossover Rate = {cor}", False)
        
def part5():
    selection_methods = [
        [tools.selTournament, {"tournsize": 3}],
        [selRank, {}],
        [tools.selRandom, {}]
    ]
    parts = ['a', 'b', 'c']
    selection_names = ["Tournament", "Rank", "Random"]

    for sel, part, name in zip(selection_methods, parts, selection_names):
        config = initial_config.copy()
        config["SELECTION_METHOD"] =  sel

        run_many(config, f"part5{part}", f"Selection Method = {name}", False)

def part6():
    run_many(optimal_config, "part6", "", True, True)

if __name__ == "__main__":
    # part1()
    # part2()
    # part3()
    # part4()
    # part5()
    part6()