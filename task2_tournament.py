###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
#from optimization_specialist_demo import tournament
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import random
from math import fabs,sqrt
import glob, os


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = "task2_tournament"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

enemies = [4,7,8] # [4,7,8] or [2,3,6]

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini = "yes")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test
selection_mode = 'tournament' 

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

"initialiseer dit????"
dom_u = 1
dom_l = -1
npop = 4
gens = 2
mutation_prob = 0.25
#############

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def crossover(x):
    parent_index = list(random.choices(range(npop), k=int(npop)))
    children = np.zeros((int(npop),n_vars))
    for i in range(0,len(parent_index),2):
        mother_index = parent_index[i]
        mother = x[mother_index,:]
        father_index = parent_index[i+1]
        father = x[father_index,:]
        alfa = np.random.uniform(0, 1)
        children[i] = alfa*mother + (1-alfa)*father
        children[i+1] = alfa*father + (1-alfa)*mother
    return children
    
def mutation(children, sigma = 1):
    for i in range(children.shape[0]):
        for j in range(0,n_vars):
            if np.random.uniform(0 , 1) <= mutation_prob:
                children[i][j] = children[i][j] + np.random.normal(0, 1)
    return children

def add_offspring(pop, offspring, fit_pop, fit_offspring): 
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)
    return pop, fit_pop

def find_best(pop, fit_pop): 
    best = np.argmax(fit_pop)
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_fit = fit_pop[best]
    return best_fit

def calc_psel(m, rank, s = 1.5):
    psel = ( (2-s) / m ) + ( ( 2*rank*(s-1) ) / ( m * (m-1)))
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
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    return chosen

def tournament(pop, fit_pop, probs):
    m = len(pop)
    winners = []
    for i in range(npop-1):
        contenders = np.random.choice(pop.shape[0], 2 , p=probs, replace=False)
        winner = battle(contenders, fit_pop)
        winners.append(winner)
        probs[winner] = 0
        probs = probs / np.sum(probs)
    return winners

def battle(contenders, fit_pop):
    fit_battle = fit_pop[contenders]
    winner = contenders[np.argmax(fit_battle)]
    return winner



# loads file with the best solution for testing
if run_mode == 'test':
    env.update_parameter('multiplemode', 'no')
    env.update_parameter('speed', 'normal')
    for run in range(1,11):
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


# initializes population loading old solutions or generating new ones

for run in range(1,3):
    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)


    # saves results for first pop
    file_aux  = open(experiment_name+'/results_{}_{}_{}.txt'.format(enemies, run, selection_mode ),'a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    ############ Evolution

    for i in range(ini_g+1, gens):
        " crossover produces offspring "
        offspring = crossover(pop)

        " mutation of arbitrary offspring "
        offspring = mutation(offspring)

        " run a single game for every offspring to get fitness "
        fit_offspring = evaluate(offspring)

        " add characteristics of offspring to population "
        pop, fit_pop = add_offspring(pop, offspring, fit_pop, fit_offspring)
        
        " calculate probability to be selected based on ranking "
        probs = select_pop(pop, fit_pop)

        " choose new population through a designed tournament "
        chosen = tournament(pop, fit_pop, probs)

        " retrieve gens best individual and manually add it to new population"
        best = np.argmax(fit_pop)   
        chosen = np.append(chosen, best)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]

        " save statistics "
        best = np.argmax(fit_pop)
        std  =  np.std(fit_pop)
        mean = np.mean(fit_pop)

        # saves results
        file_aux  = open(experiment_name+'/results_{}_{}_{}.txt'.format(enemies, run, selection_mode ),'a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen_{}_{}_{}.txt'.format(enemies, run, selection_mode ),'w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best_{}_{}_{}.txt'.format(enemies, run, selection_mode),pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()


        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()


    env.state_to_log() # checks environment state