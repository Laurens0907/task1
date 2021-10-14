import sys

from numpy.lib.nanfunctions import _nancumprod_dispatcher

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import random
from math import fabs, sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "multiple_task2"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

enemies = [7,8]  # 2 or 3 or 5
experiment_number = 0  # 10 experiments, so numbers 0 untill 9

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test
crossover_mode = 'arithmetic'  # arithmetic or one_point

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

"initialiseer dit????"
dom_u = 1
dom_l = -1
npop = 6
gens = 2
mutation_prob = 0.2

np.random.seed(420)
#############

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

############################

print('\nNEW EVOLUTION\n')

pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fit_pop = evaluate(pop)
best = np.argmax(fit_pop)
mean = np.mean(fit_pop)
std = np.std(fit_pop)
ini_g = 0
solutions = [pop, fit_pop]
env.update_solutions(solutions)

file_aux = open(experiment_name + '/results_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
    round(std, 6)))
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

#############################

for i in range(ini_g + 1, gens):
    " crossover produces offspring "
    offspring, parents = crossover(pop)

    " mutation of arbitrary offspring "
    print(offspring)
    offspring = mutation(offspring)

    " run a single game for every offspring to get fitness "
    fit_offspring = evaluate(offspring)

    "crowding"
    pop, fit_pop = crowding(pop, offspring, parents, fit_offspring, fit_pop)


    " save statistics "
    best = np.argmax(fit_pop)

    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    ##############

    # saves results
    file_aux = open(experiment_name + '/results_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name + '/gen_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), 'w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name + '/best_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()