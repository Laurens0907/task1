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


experiment_name = "individual_task1"
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
selection_mode = 'tournament'

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

"initialiseer dit????"
dom_u = 1
dom_l = -1
npop = 10
gens = 2
mutation = 0.2
last_best = 0

#############

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def crossover(x):
    "arg: population"
    "return: children"
    parent_index = list(random.sample(range(npop), int(0.8*npop)))
    children = np.zeros((int(0.4*npop),n_vars))
    f = 0
    for i in range(0,len(parent_index),2):
        mother_index = parent_index[i]
        mother = x[mother_index,:]
        father_index = parent_index[i+1]
        father = x[father_index,:]
        alfa = np.random.uniform(0,1)
        children[f] = alfa*mother + (1-alfa)*father
        # portion_mother = int(alfa*n_vars)
        # children[f] = np.append(mother[0:portion_mother],father[portion_mother:n_vars])
        f = f+1
    return children

def mutation(x, sigma = 1):
    "arg: children"
    "return: mutated children"
    # random.randint(0,len(x),replace=False)
    mutation_index = list(random.sample(range(len(x)), round(random.uniform(0,1))*len(x)))
    for i in range(len(mutation_index)):
        mutation_index_genes = list(random.sample(range(n_vars), round(random.uniform(0, 1)) * n_vars))
        for j in range(len(mutation_index_genes)):
           x[i,j] = x[i,j] + np.random.normal(0, sigma)
    return x

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

def calc_psel(m, rank, s = 1.5):
    psel = ( (2-s) / m ) + ( ( 2*rank*(s-1) ) / ( m * (m-1)))
    return psel

def select_pop(pop, fit_pop): 
    "select simpel and select door tournament"
    m = len(pop)
    probs = np.zeros(m)
    ranked_fit = np.argsort(fit_pop)
    for rank, i in enumerate(ranked_fit):
        psel = calc_psel(m, rank)
        probs[i] = psel
    "arg: fitness population, population, best"

    return probs
    "return population - worst, fitness population - worst"

def choose_pop(pop, fit_pop, probs):
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]
    return pop, fit_pop

def tournament(pop, fit_pop, probs):
    m = len(pop)
    winners = []
    for i in range(npop):
        contenders = np.random.choice(pop.shape[0], 2 , p=probs, replace=False)
        winner = battle(contenders, fit_pop)
        winners.append(winner)
        probs[winner] = 0
        probs = probs / np.sum(probs)
    pop = pop[winners]
    fit_pop = fit_pop[winners]
    return pop, fit_pop

def battle(contenders, fit_pop):
    fit_battle = fit_pop[contenders]
    winner = contenders[np.argmax(fit_battle)]
    return winner


def check_improved(current, last, not_improved): 
    "arg: best, last solution, not_improved"
    if current > last:
        not_improved = 0
    else:
        not_improved += 1
    last = current
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
 
last_mean = np.mean(fit_pop)
not_improved = 0

for i in range(ini_g+1, gens):
    #run alle methods voor gens - ini_g+1 iterations

    # crossover 
    offspring = crossover(pop)

    # mutation  -> produced kind
    offspring = mutation(offspring)

    # simulate kind om fitness te krijgen
    fit_offspring = evaluate(offspring)

    # voeg kind toe aan population
    # voeg fitness van kind toe aan population
    pop, fit_pop = add_offspring(pop, offspring, fit_pop, fit_offspring)

    # zoek beste uit population (hoogste fitness)
    # repeats best eval, for stability issues (kijk demo)
    # sla beste solution op 
    best_fit = find_best(pop, fit_pop)

    # remove slechtste uit populatie (nieuwe selectie voor populatie) ->
        # check slides hoe (eerst slechste)
    
    probs = select_pop(pop, fit_pop)

    "Change selection_mode to selection for linear ranking selection"
    if selection_mode == 'selection':
        pop, fit_pop = choose_pop(pop, fit_pop, probs)

    "Change selection_mode to tournament for tournament selection"
    if selection_mode == 'tournament':
        pop, fit_pop = tournament(pop, fit_pop, probs)


    # check of je improved op voorgaande solution
        # ja: update best solution 
        # nee: add 1 aan notimproved
    # stel last solution is x aantal keer niet improved -> kill simulations

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)

    last_mean, not_improved = check_improved(mean, last_mean, not_improved)

##############

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

    if not_improved == 10:
        break
        "is ff om error te voorkomen"

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state