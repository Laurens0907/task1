###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
from optimization_specialist_demo import tournament
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
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

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

"initialiseer dit????"
dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0

#############

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def crossover(x): 
    "arg: population"
    a = 1
    "return kind"

def mutation(x): 
    "arg: popolation, kids"
    a = 1
    "return kind na mutation"

def add_offspring(pop, offspring, fit_pop, fit_offspring): 
    "arg: population, kids, fitness pop, fitness kind"
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)
    return pop, fit_pop
    "return nieuwe population, nieuwe fitness population"

def find_best(pop, fit_pop): 
    "arg: fitness population, population"
    best = np.argmax(fit_pop)
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_fit = fit_pop[best]
    return best_fit
    "return best"

def select_pop(pop, fit_pop, best): 
    "select simpel and select door tournament"
    "arg: fitness population, population, best"
    rank_fit = np.argsort(fit_pop)
    
    return pop, fit_pop
    "return population - worst, fitness population - worst"

def check_improved(best, last, not_improved): 
    "arg: best, last solution, not_improved"
    if best > last:
        last = best
        not_improved = 0
    else:
        not_improved += 1
    return last, not_improved
    "return not_improved, last solution"
#############



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()

############ Evolution

#maak for loop
 
last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):
    error = 'weg'
    #run alle methods voor gens - ini_g+1 iterations

    # crossover 
    # mutation  -> produced kind
    # simulate kind om fitness te krijgen

    # voeg kind toe aan population
    # voeg fitness van kind toe aan population

    # zoek beste uit population (hoogste fitness)
    # repeats best eval, for stability issues (kijk demo)
    # sla beste solution op 

    # remove slechtste uit populatie (nieuwe selectie voor populatie) ->
        # check slides hoe (eerst slechste)
    
    # check of je improved op voorgaande solution
        # ja: update best solution 
        # nee: add 1 aan notimproved
    # stel last solution is x aantal keer niet improved -> kill simulations


    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


##############

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state