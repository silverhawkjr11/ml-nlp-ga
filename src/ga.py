import numpy
import random
from deap import base, creator, tools, algorithms
from joblib import load
import nltk
from nltk import pos_tag
from nltk.corpus import words

# Ensure NLTK resources are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

class GeneticAlgorithm:
    def __init__(self, model_path, dictionary_path, reduced_dict_size=10000):
        self.text_clf = load(model_path)
        self.final_dictionary = self.get_words(dictionary_path)
        self.reduced_dict = self.get_random_words(dictionary_path, reduced_dict_size)
        self.word_list = words.words()
        self.pos_dict = self.create_pos_dict()

        # Create the fitness class, which will be the base of the individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # Create the individual class based on list with fitness attribute
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def get_words(self, filename):
        with open(filename, 'r') as f:
            words = f.read().splitlines()
        return words

    def get_random_words(self, filename, n):
        with open(filename, 'r') as f:
            words = f.read().splitlines()
        if n > len(words):
            n = len(words)  # Adjust n to the size of the dictionary if n is larger
        return random.sample(words, n)

    def get_pos(self, word):
        return pos_tag([word])[0][1]

    def create_pos_dict(self):
        pos_dict = {}
        for word in self.word_list:
            pos = self.get_pos(word)
            if pos not in pos_dict:
                pos_dict[pos] = []
            pos_dict[pos].append(word)
        return pos_dict

    def fitness(self, individual):
        sentence = " ".join(individual)
        score = self.text_clf.predict([sentence])[0]
        return score,

    def mutate(self, individual):
        index = random.randrange(len(individual))
        original_word = individual[index]
        original_pos = self.get_pos(original_word)
        if original_pos in self.pos_dict:
            individual[index] = random.choice(self.pos_dict[original_pos])
        return individual,

    def run(self):
        # Create the toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.choice(self.final_dictionary), n=20)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate)
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
        # Return the best sentence found
        return f"The best sentence is: \"{sentence}\", with score: {hof[0].fitness.values[0]}"
