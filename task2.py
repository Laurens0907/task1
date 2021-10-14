###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
# from optimization_specialist_demo import tournament
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
npop = 20
gens = 10
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

# def crossover(x):
#     parent_index = list(random.choices(range(npop), k= int(npop)))
#     children = np.zeros((int(npop), n_vars))
#     f = 0
#     for i in range(0, len(parent_index), 2):
#         mother_index = parent_index[i]
#         mother = x[mother_index, :]
#         father_index = parent_index[i + 1]
#         father = x[father_index, :]
#         alfa = np.random.uniform(0, 1)
#         if crossover_mode == 'arithmetic':
#             children[f] = alfa * mother + (1 - alfa) * father
#             children[f+1] = (1-alfa) * mother + alfa * father
#         if crossover_mode == 'one_point':
#             portion_mother = int(alfa * n_vars)
#             children[f] = np.append(mother[0:portion_mother], father[portion_mother:n_vars])
#             children[f+1] = np.append(father[0:portion_mother], mother[portion_mother:n_vars])
#         f = f + 2
#     return children, parent_index


def crossover(pop):
    parent_index = []
    children = np.zeros((int(npop), n_vars))
    for i in range(0, npop, 2):
        c1 =  np.random.randint(0, npop, 2)
        c2 =  np.random.randint(0, npop, 2)
        p1 = battle(c1, fit_pop)
        p2 = battle(c2, fit_pop)
        parent_index.append(p1)
        parent_index.append(p2)
        mother = pop[p1]
        father = pop[p2]
        alfa = np.random.uniform(0, 1)
        children[i, :] = alfa * mother + (1 - alfa) * father
        children[i+1, :] = (1-alfa) * mother + alfa * father
    return children, np.array(parent_index)




def mutation(x, sigma=1):
    # random.randint(0,len(x),replace=False)
    #mutation_index = list(random.sample(range(len(x)), round(random.uniform(0, 1)) * len(x)))
    #for i in range(len(mutation_index)):
    for i in range(len(x)):
        #mutation_index_genes = list(random.sample(range(n_vars), round(random.uniform(0, 1)) * n_vars))
        for j in range(n_vars):
            if np.random.uniform(0 ,1) <= mutation_prob:
                x[i][j]= x[i][j]+np.random.normal(0, 1)
            # x[i,j] = x[i,j] + np.random.normal(0, sigma)
            # x[i,j] = limits(x[i,j])

        #x[i] = np.array(list(map(lambda y: limits(y), x[i])))
    return x

def crowding(pop,  offspring, parents, fit_offspring, fit_pop):
    pop_new = np.zeros((npop, n_vars))
    fit_pop_new =np.zeros(npop)
    for i in range(0,len(parents), 2):
        p1 = pop[parents[i],:]
        p2 = pop[parents[i+1],:]
        c1 = offspring[i]
        c2 = offspring[i+1]
        print(c1)
        print(c2)
        dist_matrix = distance_matrix([p1, p2], [c1, c2])
        couple = short_dist(dist_matrix)
        if couple == '1,1':
            winner1 = battle_crowding(fit_pop, fit_offspring, parents[i], i)
            winner2 = battle_crowding(fit_pop, fit_offspring, parents[i+1], i+1)
            if winner1 == 'parent':
                pop_new[i] = p1
                fit_pop_new[i] = fit_pop[parents[i]]
            else:
                pop_new[i] = c1
                fit_pop_new[i] = fit_offspring[i]
            if winner2 == 'parent':
                pop_new[i+1] = p2
                fit_pop_new[i+1] = fit_pop[parents[i+1]]
            else:
                pop_new[i+1] = c2
                fit_pop_new[i+1] = fit_offspring[i+1]
        else:
            winner1 = battle_crowding(fit_pop, fit_offspring, parents[i], i+1)
            winner2 = battle_crowding(fit_pop, fit_offspring, parents[i+1], i)
            if winner1 == 'parent':
                pop_new[i] = p1
                fit_pop_new[i] = fit_pop[parents[i]]
            else:
                pop_new[i] = c2
                fit_pop_new[i] = fit_offspring[i+1]
            if winner2 == 'parent':
                pop_new[i] = p2
                fit_pop_new[i+1] = fit_pop[parents[i+1]]
            else:
                pop_new[i] = c1
                fit_pop_new[i+1] = fit_offspring[i]
    return pop_new, fit_pop_new
        



def short_dist(dist):
    if dist[0][0] + dist[1][1] < dist[0][1] + dist[1][0]:
        couple = '1,1'
    else:
        couple = '1,2'
    return couple

def add_offspring(pop, offspring, fit_pop, fit_offspring):
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)
    return pop, fit_pop


def find_best(pop, fit_pop):
    best = np.argmax(fit_pop)
    fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
    best_fit = fit_pop[best]
    return best_fit


def calc_psel(m, rank, s=1.5):
    psel = ((2 - s) / m) + ((2 * rank * (s - 1)) / (m * (m - 1)))
    return psel


def select_pop(pop, fit_pop):
    m = len(pop)
    probs = np.zeros(m)
    ranked_fit = np.argsort(fit_pop)
    for rank, i in enumerate(ranked_fit):
        psel = calc_psel(m, rank)
        probs[i] = psel
    return probs


def choose_pop(pop, fit_pop, probs):
    chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
    return chosen


def tournament(pop, fit_pop, probs):
    m = len(pop)
    winners = []
    for i in range(npop - 1):
        contenders = np.random.choice(pop.shape[0], 2, p=probs, replace=False)
        winner = battle(contenders, fit_pop)
        winners.append(winner)
        probs[winner] = 0
        probs = probs / np.sum(probs)
    return winners

def battle_crowding(fit_pop, fit_offspring, parent, child):
    fit_parent = fit_pop[parent]
    fit_child = fit_offspring[child]
    if fit_parent > fit_child:
        winner = 'parent'
    else:
        winner = 'child'
    return winner


def battle(contenders, fit_pop):
    fit_battle = fit_pop[contenders]
    winner = contenders[np.argmax(fit_battle)]
    return winner


def check_improved(current, last, not_improved):
    if current > last:
        not_improved = 0
    else:
        not_improved += 1
    last = current
    return last, not_improved

def distance_matrix(parents, children):
    distance_matrix = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            distance_matrix[i][j] = np.sqrt(np.sum((parents[i] - children[j])**2))
    return distance_matrix

#############


# loads file with the best solution for testing
if run_mode == 'test':
    file_aux = open(experiment_name + '/boxplot_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), 'a')
    for test_run in range(5):
        bsol = np.loadtxt(experiment_name + '/best_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode))
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        # fitness = evaluate([bsol])[0]
        f, p, e, t = env.play(pcont=bsol)
        gain = p - e
        file_aux.write(str(gain) + '\n')
    file_aux.close()

    sys.exit(0)

# initializes population loading old solutions or generating new ones

print('\nNEW EVOLUTION\n')

pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fit_pop = evaluate(pop)
best = np.argmax(fit_pop)
mean = np.mean(fit_pop)
std = np.std(fit_pop)
ini_g = 0
solutions = [pop, fit_pop]
env.update_solutions(solutions)

# if not os.path.exists(experiment_name+'/evoman_solstate'):

#     print( '\nNEW EVOLUTION\n')

#     pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
#     fit_pop = evaluate(pop)
#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)
#     ini_g = 0
#     solutions = [pop, fit_pop]
#     env.update_solutions(solutions)

# else:

#     print( '\nCONTINUING EVOLUTION\n')

#     env.load_state()
#     pop = env.solutions[0]
#     fit_pop = env.solutions[1]

#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)

#     # finds last generation number
#     file_aux  = open(experiment_name+'/gen.txt','r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()


# saves results for first pop
file_aux = open(experiment_name + '/results_{}_{}_{}.txt'.format(enemies, experiment_number, crossover_mode), 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
    round(std, 6)))
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

############ Evolution


last_mean = np.mean(fit_pop)
not_improved = 0

for i in range(ini_g + 1, gens):
    " crossover produces offspring "
    offspring, parents = crossover(pop)

    " mutation of arbitrary offspring "
    offspring = mutation(offspring)

    " run a single game for every offspring to get fitness "
    fit_offspring = evaluate(offspring)

    " add characteristics of offspring to population "
    #pop, fit_pop = add_offspring(pop, offspring, fit_pop, fit_offspring)

    "crowding"
    pop, fit_pop = crowding(pop, offspring, parents, fit_offspring, fit_pop)

    " calculate probability to be selected based on ranking "
   # probs = select_pop(pop, fit_pop)

    " choose new population through a designed tournament "
    #chosen = tournament(pop, fit_pop, probs)

    " retrieve gens best individual and manually add it to new population"
    # best = np.argmax(fit_pop)
    # chosen = np.append(chosen, best)
    # pop = pop[chosen]
    # fit_pop = fit_pop[chosen]

    " save statistics "
    best = np.argmax(fit_pop)

    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    " check if mean has improved "
    last_mean, not_improved = check_improved(mean, last_mean, not_improved)

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

    if not_improved == 10:
        break
        "is ff om error te voorkomen"

fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state