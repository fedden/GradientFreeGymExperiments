import os
import json
import argparse
import datetime
import numpy as np
import multiprocessing
from optimisers import *
from utils import user_query
from simulation import GymSimulation


parser = argparse.ArgumentParser(description='Optimiser for various natural computing methods to parameterise neural networks')
parser.add_argument('-l', '--layer_sizes', help='Sequence of values for the hidden dimensions', nargs='+', type=int, required=True)
parser.add_argument('-e', '--env_name', help='OpenAI gym environment name', type=str, required=True)
parser.add_argument('-r', '--rounds_optimisation', help='How many rounds of optimisation', type=int, required=True)
parser.add_argument('-p', '--population_size', help='Approximate size of the population', type=int, required=True)
parser.add_argument('-a', '--amount_experiments', help='Amount of experiment repeats', type=int, required=False)
parser.add_argument('-n', '--number_repeats', help='Number of environment repeats', type=int, required=False)
parser.add_argument('-s', '--seed', help='PRNG seed', type=int, required=False)
parser.add_argument('-i', '--invert_reward', help='', required=False)
parser.add_argument('-o', '--reward_offset', help='', type=int, required=False)
parser.add_argument('-m', '--mean', help='', type=float, required=False)
parser.add_argument('-d', '--deviation', help='', type=float, required=False)
parser.add_argument('-f', '--load_from_folder', help='', type=str, required=False)


args = vars(parser.parse_args())

if args['amount_experiments'] is None:
    args['amount_experiments'] = 1
if args['number_repeats'] is None:
    args['number_repeats'] = 30
if args['seed'] is None:
    args['seed'] = 8
if args['invert_reward'] is None:
    args['invert_reward'] = True
if args['reward_offset'] is None:
    args['reward_offset'] = 0.0
if args['mean'] is None:
    args['mean'] = 0.0
if args['deviation'] is None:
    args['deviation'] = 0.01
if args['load_from_folder'] is None:
    args['load_from_folder'] = ''

np.random.seed(args['seed'])

contenders = ["rdm", "de", "dfo", "ga", "hy1", "hy2", "hy3", "hy4"]
win_counts = {}
for contender in contenders:
    win_counts[contender] = 0

simulator = GymSimulation(args['layer_sizes'],
                          args['env_name'],
                          seed=args['seed'],
                          number_repeats=args['number_repeats'],
                          invert_reward=args['invert_reward'],
                          reward_offset=args['reward_offset'])

while args['population_size'] % 3 != 0:
    args['population_size'] += 1

population_shape = (args['population_size'], simulator.get_solution_size())
all_experiments_fitnesses_histories = []

continue_string = "\n\n{} members of the population, each of size {}, will be optimised over {} threads. Continue?"
continue_string = continue_string.format(args['population_size'],
                                         simulator.get_solution_size(),
                                         multiprocessing.cpu_count())
do_continue = user_query(continue_string)

if do_continue:

    root_directory = args['env_name'] + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(root_directory):
        os.makedirs(root_directory)

    for experiment in range(args['amount_experiments']):

        if args['load_from_folder'] is not '':
            rdm_population = np.load(os.path.join(args['load_from_folder'], 'rdm_population'))
            de_population = np.load(os.path.join(args['load_from_folder'], 'de_population'))
            ga_population = np.load(os.path.join(args['load_from_folder'], 'dfo_population'))
            dfo_population = np.load(os.path.join(args['load_from_folder'], 'ga_population'))
            hybrid_1_population = np.load(os.path.join(args['load_from_folder'], 'hybrid_1_population'))
            hybrid_2_population = np.load(os.path.join(args['load_from_folder'], 'hybrid_2_population'))
            hybrid_3_population = np.load(os.path.join(args['load_from_folder'], 'hybrid_3_population'))
            hybrid_4_population = np.load(os.path.join(args['load_from_folder'], 'hybrid_4_population'))
        else:
            population = np.random.normal(size=population_shape,
                                          loc=args['mean'],
                                          scale=args['deviation'])
            rdm_population = np.copy(population)
            de_population = np.copy(population)
            ga_population = np.copy(population)
            dfo_population = np.copy(population)
            hybrid_1_population = np.copy(population)
            hybrid_2_population = np.copy(population)
            hybrid_3_population = np.copy(population)
            hybrid_4_population = np.copy(population)

        print("started            ", end="\r")

        experiment_directory = os.path.join(root_directory, "experiment_{}".format(experiment))
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)

        if args['load_from_folder'] is not '':
            experiment_data_path = os.path.join(os.path.join(args['load_from_folder'], "experiment_data.json")
            with open(experiment_data_path) as json_data:
                experiment_data = json.load(json_data)
            begin = experiment_data['iteration']
            experiment_fitnesses = experiment_data['scores']
        else:
            experiment_data = {}
            begin = 0
            start_fitnesses = simulator.get_fitnesses(population)
            start_fitness = np.amin(start_fitnesses)
            experiment_fitnesses = [np.array([start_fitness for _ in range(len(contenders))])]

        end = args['rounds_optimisation']
        for iteration in range(begin, end):

            rdm_population, rdm_best, rdm_best_fitness = \
                optimise_rdm(rdm_population, args['mean'], args['deviation'], simulator)
            print("completed rdm        ", end="\r")

            de_population, de_best, de_best_fitness = \
                optimise_de(de_population, args['mean'], args['deviation'], simulator)
            print("completed de         ", end="\r")

            dfo_population, dfo_best, dfo_best_fitness = \
                optimise_dfo(dfo_population, args['mean'], args['deviation'], simulator)
            print("completed dfo        ", end="\r")

            ga_population, ga_best, ga_best_fitness = \
                optimise_ga(ga_population, args['mean'], args['deviation'], simulator)
            print("completed ga         ", end="\r")

            hybrid_1_population, hybrid_1_best, hybrid_1_best_fitness = \
                optimise_hybrid(hybrid_1_population, args['mean'], args['deviation'], simulator, update_method=1)
            print("completed hy1        ", end="\r")

            hybrid_2_population, hybrid_2_best, hybrid_2_best_fitness = \
                optimise_hybrid(hybrid_2_population, args['mean'], args['deviation'], simulator, update_method=2)
            print("completed hy2        ", end="\r")

            hybrid_3_population, hybrid_3_best, hybrid_3_best_fitness = \
                optimise_hybrid(hybrid_3_population, args['mean'], args['deviation'], simulator, update_method=3)
            print("completed hy3        ", end="\r")

            hybrid_4_population, hybrid_4_best, hybrid_4_best_fitness = \
                optimise_hybrid(hybrid_4_population, args['mean'], args['deviation'], simulator, update_method=4)
            print("completed hy4        ", end="\r")

            scores = np.array([
                rdm_best_fitness,
                de_best_fitness,
                dfo_best_fitness,
                ga_best_fitness,
                hybrid_1_best_fitness,
                hybrid_2_best_fitness,
                hybrid_3_best_fitness,
                hybrid_4_best_fitness
            ])

            experiment_fitnesses.append(scores)

            experiment_data['iteration'] = iteration
            experiment_data['scores'] = [e.tolist() for e in experiment_fitnesses]
            experiment_data_save_path = os.path.join(experiment_directory, "experiment_data.json")
            with open(experiment_data_save_path, 'w') as fp:
                json.dump(experiment_data, fp)
            np.save(os.path.join(experiment_directory, 'rdm_population'), rdm_population)
            np.save(os.path.join(experiment_directory, 'de_population'), de_population)
            np.save(os.path.join(experiment_directory, 'dfo_population'), dfo_population)
            np.save(os.path.join(experiment_directory, 'ga_population'), ga_population)
            np.save(os.path.join(experiment_directory, 'hybrid_1_population'), hybrid_1_population)
            np.save(os.path.join(experiment_directory, 'hybrid_2_population'), hybrid_2_population)
            np.save(os.path.join(experiment_directory, 'hybrid_3_population'), hybrid_3_population)
            np.save(os.path.join(experiment_directory, 'hybrid_4_population'), hybrid_4_population)

            best_indices = np.where(scores == scores.min())[0].tolist()
            best_contenders = [contenders[index] for index in best_indices]

            amount_done = int((iteration + 1) / args['rounds_optimisation'] * 100)
            progress_string = "{}% done, best: {}, score: {}                     "
            print(progress_string.format(amount_done, best_contenders, np.sort(scores)))

            if len(best_contenders) == len(contenders):
                break

        print("\nExperiment", experiment, "results:")
        for i in range(len(scores)):
            print(contenders[i], "-", scores[i])
        print()

        all_experiments_fitnesses_histories.append(experiment_fitnesses)
else:
    print("Cancelling optimisation. No search has taken place.")
