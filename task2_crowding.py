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

experiment_name = "task2_crowding"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

enemies = [4,7,8]  # [4,7,8] or [2,3,6]
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
selection_mode = 'crowding'  # crowding or tournament

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

"initialiseer dit"
dom_u = 1
dom_l = -1
npop = 60  #Moet deelbaar zijn door 4!!
gens = 20
offspring_per_couple = 2  #Moet 2 blijven!!!
mutation_prob = 0.25
runs = 10

np.random.seed(84088)
#############

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


############################
# loads file with the best solution for testing
if run_mode == 'test':
    env.update_parameter('multiplemode', 'no')
    env.update_parameter('speed', 'normal')
    for run in range(1,runs+1):
        file_aux = open(experiment_name + '/boxplot_{}_{}_{}.txt'.format(enemies, run, selection_mode), 'a')
        for i in range(1,9):
            env.update_parameter('enemies', [i])
            for test_run in range(5):
                bsol = np.loadtxt(experiment_name + '/best_{}_{}_{}.txt'.format(enemies, run, selection_mode))
                print('\n RUNNING SAVED BEST SOLUTION \n')
                f, p, e, t = env.play(pcont=bsol)
                gain = p - e
                file_aux.write(str(gain) + '\n')
        file_aux.close()

    sys.exit(0)


##############################################################################################
def battle(pop,contender1,contender2):
    if fit_pop[contender1] > fit_pop[contender2]:
        return pop[contender1][0],contender1
    else:
        return pop[contender2][0],contender2

def crossover(pop):
    new_offspring = np.zeros((0,n_vars))
    parents_index = np.zeros(n_vars,dtype=int)
    # probs = fit_pop/np.sum(fit_pop)
    for i in range(0,pop.shape[0], 2):
        # contender1 = np.random.randint(0,pop.shape[0], 1)
        # contender2 = np.random.randint(0,pop.shape[0], 1)
        # contender3 = np.random.randint(0,pop.shape[0], 1)
        # contender4 = np.random.randint(0,pop.shape[0], 1)
        # parent1,index_parent1 = battle(pop,contender1,contender2)
        # parent2,index_parent2 = battle(pop,contender3,contender4)
        index_parent1 = np.random.randint(0, pop.shape[0], 1)
        index_parent2 = np.random.randint(0, pop.shape[0], 1)
        parent1 = pop[index_parent1][0]
        parent2 = pop[index_parent2][0]
        parents_index[i] = index_parent1
        parents_index[i+1] = index_parent2
        children = np.zeros((offspring_per_couple, n_vars))

        for j in range(0,offspring_per_couple,2):
            alfa = np.random.uniform(0 , 1)
            children[j] = alfa * parent1 + (1 - alfa) * parent2
            children[j+1] = alfa * parent2 + (1 - alfa) * parent1

            for k in range(0,n_vars):
                if np.random.uniform(0 , 1) <= mutation_prob:
                    children[j][k] = children[j][k] + np.random.normal(0, 1)
                if np.random.uniform(0, 1) <= mutation_prob:
                    children[j+1][k] = children[j+1][k] + np.random.normal(0, 1)

            new_offspring = np.vstack((new_offspring, children[j]))
            new_offspring = np.vstack((new_offspring, children[j+1]))
    return new_offspring,parents_index

def crowding_couple(fit_parent1,fit_parent2,fit_child1,fit_child2):
    if np.abs(fit_parent1 - fit_child1) + np.abs(fit_parent2 - fit_child2) <= np.abs(fit_parent1 - fit_child2) + np.abs(fit_parent2 - fit_child1):
        return True
    else:
        return False


def crowding(pop,offspring,parents_index,fit_offspring,fit_pop):
    new_pop = np.zeros((0,n_vars))
    new_fit_pop = np.zeros(npop)
    for i in range(0,len(offspring),2):
        child1 = offspring[i]
        fit_child1 = fit_offspring[i]
        child2 = offspring[i+1]
        fit_child2 = fit_offspring[i+1]
        parent1 = pop[parents_index[i]]
        fit_parent1 = fit_pop[parents_index[i]]
        parent2 = pop[parents_index[i+1]]
        fit_parent2 = fit_pop[parents_index[i+1]]
        if crowding_couple(fit_parent1,fit_parent2,fit_child1,fit_child2):
            if fit_parent1 > fit_child1:
                new_pop = np.vstack((new_pop, parent1))
                new_fit_pop[i] = fit_parent1
            else:
                new_pop = np.vstack((new_pop, child1))
                new_fit_pop[i] = fit_child1
            if fit_parent2 > fit_child2:
                new_pop = np.vstack((new_pop, parent2))
                new_fit_pop[i+1] = fit_parent2
            else:
                new_pop = np.vstack((new_pop, child2))
                new_fit_pop[i+1] = fit_child2
        else:
            if fit_parent1 > fit_child2:
                new_pop = np.vstack((new_pop, parent1))
                new_fit_pop[i] = fit_parent1
            else:
                new_pop = np.vstack((new_pop, child2))
                new_fit_pop[i] = fit_child2
            if fit_parent2 > fit_child1:
                new_pop = np.vstack((new_pop, parent2))
                new_fit_pop[i+1] = fit_parent2
            else:
                new_pop = np.vstack((new_pop, child1))
                new_fit_pop[i+1] = fit_child1
    return new_pop,new_fit_pop
##############################################################################################

for run in range(1,runs+1):
    print('\nNEW EVOLUTION\n')

    pop_best = np.zeros((0,n_vars))
    fit_pop_best = np.zeros(0)
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    pop_best = np.vstack((pop_best,pop[best]))
    fit_pop_best = np.append(fit_pop_best, fit_pop[best])

    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    file_aux = open(experiment_name + '/results_{}_{}_{}.txt'.format(enemies, run, selection_mode), 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    for i in range(ini_g + 1, gens):
        " crossover produces offspring "
        offspring, parents_index = crossover(pop)

        " run a single game for every offspring to get fitness "
        fit_offspring = evaluate(offspring)

        "crowding"
        pop, fit_pop = crowding(pop, offspring, parents_index, fit_offspring, fit_pop)

        " save statistics "
        best = np.argmax(fit_pop)
        std = np.std(fit_pop)
        mean = np.mean(fit_pop)
        pop_best = np.vstack((pop_best, pop[best]))
        fit_pop_best = np.append(fit_pop_best,fit_pop[best])
        overall_best = np.argmax(fit_pop_best)
        ##############

        # saves results
        file_aux = open(experiment_name + '/results_{}_{}_{}.txt'.format(enemies, run, selection_mode), 'a')
        print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))
        file_aux.write(
            '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
        file_aux.close()

        # saves generation number
        file_aux = open(experiment_name + '/gen_{}_{}_{}.txt'.format(enemies, run, selection_mode), 'w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name + '/best_{}_{}_{}.txt'.format(enemies, run, selection_mode), pop_best[overall_best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    env.state_to_log()  # checks environment state