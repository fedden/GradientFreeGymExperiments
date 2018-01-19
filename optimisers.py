import numpy as np


def optimise_dfo(population,
                 mean,
                 std,
                 simulator,
                 disturbance_threshold=0.1,
                 print_results=False):

    population_size = len(population)
    solution_size = simulator.get_solution_size()

    fitnesses = simulator.get_fitnesses(population)

    swarms_best_index = np.argmin(fitnesses)
    best_member = population[swarms_best_index]

    if print_results:
        print("index:", swarms_best_index,
              " fitness:", fitnesses[swarms_best_index])

    population_up = np.roll(population, -1, axis=0)
    population_down = np.roll(population, 1, axis=0)

    fitnesses_repeated = np.repeat(fitnesses.reshape(-1, 1),
                                   solution_size, axis=1)
    fitnesses_up = np.roll(fitnesses_repeated, -1, axis=0)
    fitnesses_down = np.roll(fitnesses_repeated, 1, axis=0)

    best_neighbours = np.where(fitnesses_up < fitnesses_down,
                               population_up, population_down)

    disturbance_rolls = np.random.uniform(size=(population_size, solution_size))

    random_resets = np.random.normal(size=(population_size, solution_size),
                                     loc=mean, scale=std)
    move_amount = np.random.uniform(size=(population_size, solution_size))
    fly_update = best_neighbours + move_amount * (best_member - best_neighbours)

    population = np.where(disturbance_rolls < disturbance_threshold,
                          random_resets, fly_update)
    return population, best_member, fitnesses[swarms_best_index]


def optimise_rdm(old_population, mean, std, simulator):

    solution_size = simulator.get_solution_size()
    new_population = np.random.normal(size=old_population.shape,
                                      loc=mean, scale=std)
    new_fitnesses = simulator.get_fitnesses(new_population)
    old_fitnesses = simulator.get_fitnesses(old_population)

    if np.amin(new_fitnesses) < np.amin(old_fitnesses):
        best_index = np.argmin(new_fitnesses)
        best_fitness = np.amin(new_fitnesses)
        best_member = new_population[best_index]
    else:
        best_index = np.argmin(old_fitnesses)
        best_fitness = np.amin(old_fitnesses)
        best_member = old_population[best_index]

    old_fitnesses = np.repeat(old_fitnesses.reshape(-1, 1),
                              solution_size, axis=1)
    new_fitnesses = np.repeat(new_fitnesses.reshape(-1, 1),
                              solution_size, axis=1)

    return np.where(old_fitnesses < new_fitnesses, old_population, new_population), best_member, best_fitness


def optimise_de(population,
                mean,
                std,
                simulator,
                print_results=False,
                differential_weight=1.0,
                crossover_probability=0.5):

    population_size = len(population)
    solution_size = simulator.get_solution_size()

    random_1 = np.arange(0, population_size)
    np.random.shuffle(random_1)
    random_2 = np.arange(0, population_size)
    np.random.shuffle(random_2)
    random_3 = np.arange(0, population_size)
    np.random.shuffle(random_3)
    random_trio_1 = np.reshape(population[random_1], newshape=(-1, 3, solution_size))
    random_trio_2 = np.reshape(population[random_2], newshape=(-1, 3, solution_size))
    random_trio_3 = np.reshape(population[random_3], newshape=(-1, 3, solution_size))

    mutation_trios = np.concatenate((random_trio_1, random_trio_2, random_trio_3))

    vectors_1, vectors_2, vectors_3 = np.split(mutation_trios, axis=1, indices_or_sections=3)

    doners = vectors_1 + differential_weight * (vectors_2 - vectors_3)
    doners = doners.reshape((-1, solution_size))

    crossover_probabilities = np.random.uniform(size=(population_size, solution_size))

    trial_population = np.where(crossover_probabilities < crossover_probability, doners, population)

    trial_fitnesses = simulator.get_fitnesses(trial_population)
    og_fitnesses = simulator.get_fitnesses(population)

    if np.amin(trial_fitnesses) < np.amin(og_fitnesses):
        best_index = np.argmin(trial_fitnesses)
        best_fitness = np.amin(trial_fitnesses)
        best_member = trial_population[best_index]
    else:
        best_index = np.argmin(og_fitnesses)
        best_fitness = np.amin(og_fitnesses)
        best_member = population[best_index]

    trial_fitnesses = np.repeat(trial_fitnesses.reshape(-1, 1), solution_size, axis=1)

    og_fitnesses = np.repeat(og_fitnesses.reshape(-1, 1), solution_size, axis=1)

    if print_results:
        print("index:", best_index, " fitness:", og_fitnesses[best_index][0])

    return np.where(trial_fitnesses < og_fitnesses, trial_population, population), best_member, best_fitness


def optimise_ga(population,
                mean,
                std,
                simulator,
                print_results=False,
                mutation_rate=0.1,
                elitism=0.1):

    population_size = len(population)
    solution_size = simulator.get_solution_size()
    elitism_amount = int(elitism * population_size)

    fitnesses = simulator.get_fitnesses(population)

    best_index = np.argmin(fitnesses)
    best_member = population[best_index]

    parents_a_indices = np.random.randint(population_size, size=population_size)
    parents_b_indices = np.random.randint(population_size, size=population_size)
    parents_a = population[parents_a_indices]
    parents_b = population[parents_b_indices]

    crossover_probabilities = np.random.uniform(size=(population_size, solution_size))

    child_population = np.where(crossover_probabilities < 0.5,
                                parents_a,
                                parents_b)

    mutation_probabilities = np.random.uniform(size=(population_size, solution_size))
    mutation_change = np.random.normal(size=(population_size, solution_size), loc=mean, scale=std)
    child_population = np.where(mutation_probabilities < mutation_rate,
                                child_population + mutation_change,
                                child_population)
    if print_results:
        print("index:", best_index, " fitness:", fitnesses[best_index])

    sort_indices = np.argsort(fitnesses)
    sorted_population = population[sort_indices]

    return np.concatenate((sorted_population[:elitism_amount], child_population[elitism_amount:])), best_member, fitnesses[best_index]


def optimise_hybrid(population,
                    mean,
                    std,
                    simulator,
                    print_results=False,
                    crossover_probability=0.5,
                    update_method=3):

    solution_size = simulator.get_solution_size()

    og_fitnesses = simulator.get_fitnesses(population)

    best_fly_index = np.argmin(og_fitnesses)
    best_fly = population[best_fly_index]
    if print_results: print(best_fly)

    population_size = len(population)
    shape = (population_size, solution_size)
    random_1 = np.arange(0, population_size)
    np.random.shuffle(random_1)
    random_2 = np.arange(0, population_size)
    np.random.shuffle(random_2)
    random_3 = np.arange(0, population_size)
    np.random.shuffle(random_3)
    random_trio_1 = population[random_1].reshape((-1, 3, solution_size))
    random_trio_2 = population[random_2].reshape((-1, 3, solution_size))
    random_trio_3 = population[random_3].reshape((-1, 3, solution_size))

    shuffled_populations = (random_trio_1, random_trio_2, random_trio_3)

    mutation_trios = np.concatenate(shuffled_populations, axis=0)

    vectors_1, vectors_2, vectors_3 = tuple(np.split(mutation_trios,
                                                     axis=1,
                                                     indices_or_sections=3))
    if update_method == 1:
        doners = vectors_1 + 1.0 * (best_fly - vectors_1)

    elif update_method == 2:
        dispersion_amount = np.random.uniform(size=(population_size))
        dispersion_amount = np.repeat(dispersion_amount.reshape((-1, 1)),
                                      solution_size, axis=1)
        dispersion_amount = dispersion_amount.reshape(shape)
        flies = ((vectors_1 + vectors_2 + vectors_3) / 3.0).reshape(shape)
        doners = flies + dispersion_amount * (best_fly - flies)

    elif update_method == 3:
        flies = ((vectors_1 + vectors_2 + vectors_3) / 3.0).reshape(shape)
        doners = flies + (best_fly - flies)

    elif update_method == 4:
        dispersion_amount = np.random.uniform(size=(population_size))
        dispersion_amount = np.repeat(dispersion_amount.reshape((-1, 1)),
                                      solution_size, axis=1)
        dispersion_amount = dispersion_amount.reshape(shape)
        random_fit_1 = og_fitnesses[random_1].reshape((-1, 3, 1))
        random_fit_2 = og_fitnesses[random_1].reshape((-1, 3, 1))
        random_fit_3 = og_fitnesses[random_1].reshape((-1, 3, 1))
        shuffled_fitnesses = (random_fit_1, random_fit_2, random_fit_3)
        fitness_trios = np.concatenate(shuffled_fitnesses, axis=0)
        fitness_1, fitness_2, fitness_3 = tuple(np.split(fitness_trios,
                                                         axis=1,
                                                         indices_or_sections=3))
        fitness_1 = np.repeat(fitness_1.reshape((-1, 1)), solution_size, axis=1).reshape((-1, 1, solution_size))
        fitness_2 = np.repeat(fitness_2.reshape((-1, 1)), solution_size, axis=1).reshape((-1, 1, solution_size))
        fitness_3 = np.repeat(fitness_3.reshape((-1, 1)), solution_size, axis=1).reshape((-1, 1, solution_size))
        flies = np.where(fitness_1 < fitness_2, vectors_1, vectors_2)
        flies = np.where((fitness_3 < fitness_1) & (fitness_3 < fitness_2),
                         vectors_3, flies)
        flies = flies.reshape(shape)
        doners = flies + dispersion_amount * (best_fly - flies)

    doners = doners.reshape(shape)


    crossover_probabilities = np.random.uniform(size=shape)
    do_crossover = crossover_probabilities < crossover_probability
    trial_population = np.where(do_crossover, doners, population)

    if not update_method == 2:
        disturb_probabilities = np.random.uniform(size=shape)
        disturb_threshold = 0.1
        do_disturb = disturb_probabilities < disturb_threshold
        resets = np.random.normal(size=(population_size, solution_size), loc=mean, scale=std)
        trial_population = np.where(do_disturb, resets, trial_population)

    trial_fitnesses = simulator.get_fitnesses(trial_population)

    if np.amin(trial_fitnesses) < np.amin(og_fitnesses):
        best_index = np.argmin(trial_fitnesses)
        best_fitness = np.amin(trial_fitnesses)
        best_member = trial_population[best_index]
    else:
        best_index = np.argmin(og_fitnesses)
        best_fitness = np.amin(og_fitnesses)
        best_member = population[best_index]

    trial_fitnesses = np.repeat(trial_fitnesses.reshape((-1, 1)),
                                solution_size, axis=1)

    og_fitnesses = np.repeat(og_fitnesses.reshape((-1, 1)),
                             solution_size, axis=1)
    return np.where(trial_fitnesses < og_fitnesses, trial_population, population), best_member, best_fitness
