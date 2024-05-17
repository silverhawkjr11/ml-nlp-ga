import numpy
import random
from deap import base, creator, tools, algorithms
from joblib import load

# Create the fitness class, which will be the base of the individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
text_clf = load('best_model.joblib')

# Create the individual class based on list with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMax)
def get_words(filename):
    with open(filename, 'r') as f:
        words = f.read().splitlines()
    return words

final_dictionary = get_words("new_dictionary.txt")
import random

def get_random_words(filename, n):
    with open(filename, 'r') as f:
        words = f.read().splitlines()
    return random.sample(words, n)

# Usage
reduced_dict = get_random_words('dictionary.txt', 10000)
# Define the fitness function
def fitness(individual):
    sentence = " ".join(individual)
    score = text_clf.predict([sentence])[0]
    return score,

# Define the mutation function
def mutate(individual):
    index = random.randrange(len(individual))
    individual[index] = random.choice(reduced_dict)
    return individual,
def ga():
    # Create the toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.choice(final_dictionary), n=20)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize a population
    pop = toolbox.population(n=300)

    # Define the hall of fame
    hof = tools.HallOfFame(1)

    # Perform the genetic algorithm
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    sentence = " ".join(hof[0])
    # Print the best sentence found
    return (f"the best sentence is: \"{sentence}\", with score: {hof[0].fitness.values}")