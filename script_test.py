# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:00:05 2023

@author: luis.pinos-ullauri
"""

from scipy.special import expit
import read_functions as rdf
import math
import os
import pygad
import random as rand
import numpy as np
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



############### User-defined functions for the Genetic Algorithm ##############

########## Estimation of the fitness and score functions

### Function that returns the overall batch fitness of the list solutions across all soft skill dimensions
### It checks whether the fitness amongst the different dimensions is compensated or not
### as well as which scoring function to use

def fitness_func_batch(ga_instance, solutions, solutions_indices):
    batch_fitness = []
    soft_skill_scores=[]
    for i in range(len(solutions)):
        solution=solutions[i]   
        soft_skill_scores_solution=[]
        if compensatory:
            fitness=0
        else:
            fitness=1
        for i in range(10):#soft skill id from 0 to 9
            estimated_outcome=soft_skill_estimation_mean(solution,i)
            if score_function==1:
                score=linear(estimated_outcome,i)
            elif score_function==2:
                score=logistic(estimated_outcome,i)
            elif score_function==3:
                score=quadratic(estimated_outcome,i)
            if ga_instance is None:
                soft_skill_scores_solution.append(score)            
            if compensatory:
                fitness=fitness+score*(1/10)
            else:
                fitness=fitness*score
        batch_fitness.append(fitness)
        if ga_instance is None:
            soft_skill_scores.append(soft_skill_scores_solution)
    if ga_instance is None:
        return batch_fitness,soft_skill_scores
    else:
        return batch_fitness

    
### Function that returns the overall fitness of the solution across all soft skill dimensions
### It checks whether the fitness amongst the different dimensions is compensated or not
### as well as which scoring function to use
def fitness_func(ga_instance, solution, solution_idx):
    soft_skill_scores=[]
    if compensatory:
        fitness=0
    else:
        fitness=1
    for i in range(10):#soft skill id from 0 to 9
        estimated_outcome=soft_skill_estimation_mean(solution,i)
        if score_function==1:
            score=linear(estimated_outcome,i)
        elif score_function==2:
            score=logistic(estimated_outcome,i)
        elif score_function==3:
            score=quadratic(estimated_outcome,i)
        if ga_instance is None:
            soft_skill_scores.append(score)            
        if compensatory:
            fitness=fitness+score*(1/10)
        else:
            fitness=fitness*score
    if ga_instance is None:
        return fitness,soft_skill_scores
    return fitness

### Linear scoring function
### Fair scoring with no bonuses nor penalisations if the
### soft skill proficiency mean reaches or not the expected profile
### Scoring function rescaled so that it ranges from 0 to 1
def linear(estimated_skill,soft_skill_id):
    #function min value
    f_min=min_skill-desired_outcome[soft_skill_id]
    #function max value
    f_max=max_skill-desired_outcome[soft_skill_id]    
    return (estimated_skill-desired_outcome[soft_skill_id]-f_min)/(f_max-f_min)
   
### Logistic scoring function
### Stricter scoring with penalisations and bonuses if the
### soft skill proficiency mean reaches or not the expcted profile
### Scoring function rescaled so that it ranges from 0 to 1     
def logistic(estimated_skill,soft_skill_id):
    #crossing value with linear function at estimated_goal=goal_skill
    fcrossing=(desired_outcome[soft_skill_id]-min_skill)/(max_skill-min_skill)
    return (1)/(1+((1-fcrossing)/fcrossing)*pow(math.e,3*(desired_outcome[soft_skill_id]-estimated_skill)))

### Quadratic Root scoring function
### Less demanding scoring that allows an easier scoring if the
### soft skill proficiency mean reaches or not the expcted profile
### Scoring function rescaled so that it ranges from 0 to 1  
def quadratic(estimated_skill,soft_skill_id):
    #crossing value with linear function at estimated_goal=goal_skill
    fcrossing=(desired_outcome[soft_skill_id]-min_skill)/(max_skill-min_skill)
    #function min value
    f_min=-1*pow(min_skill-desired_outcome[soft_skill_id],2)
    #function max value
    f_max=1*pow(max_skill-desired_outcome[soft_skill_id],2)
    if estimated_skill<=desired_outcome[soft_skill_id]:            
        return -1*pow(estimated_skill-desired_outcome[soft_skill_id],2)/(-f_min)*fcrossing+fcrossing
    else:
        return 1*pow(estimated_skill-desired_outcome[soft_skill_id],2)/(f_max)*(1-fcrossing)+fcrossing

### Function that estimates the soft skill mean based on the ordinal logistic regression model
### It calculates the probability of each level and estimates the mean SUM x*P(X=x)
def soft_skill_estimation_mean(solution,soft_skill_id):
    linear_combination=0
    for i in range(len(solution)):
        if solution[i]!=0:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]]
    linear_combination=linear_combination+theta[soft_skill_id]
    eta12=thresholds.iloc[soft_skill_id,0]-linear_combination
    eta23=thresholds.iloc[soft_skill_id,1]-linear_combination
    eta34=thresholds.iloc[soft_skill_id,2]-linear_combination
    p_1=expit(eta12)
    p_2=expit(eta23)-p_1
    p_3=expit(eta34)-p_1-p_2
    p_4=1-p_3-p_2-p_1
    expected_outcome=1*p_1+2*p_2+3*p_3+4*p_4
    return expected_outcome


########## Genetic Algorithm user-defined functions
   
### Function that performs uniform crossover between two parent combinations
### maintaining the constraint of different followed courses
def unifcrossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        child=parent2.copy()#child is copy of parent2 by default
        if np.random.random()<=ga_instance.crossover_probability:
            for i in range(len(parent1)):
                if parent1[i] not in parent2:#gene parent1[i] not in parent[2]
                    if rand.random()<0.5:#copy parent1                        
                        child[i]=parent1[i]         
        offspring.append(child)
        idx += 1
    return np.array(offspring)


### Function that performs a single point mutation on one course
### It checks whether the new course is already in the combination and if so
### Generates a new one
def mutation_func(offspring, ga_instance):
    view = offspring.shape[0]
    for chromosome_idx in range(view):
        available_course_for_sol = np.setdiff1d(possible_courses,offspring[chromosome_idx]) #Remove all course of indiv in the catalog to draw a rn value in
        if len(available_course_for_sol)>=1:
            random_gene=np.random.choice(available_course_for_sol)
            random_gene_idx = np.random.choice(range(offspring.shape[1]))
            offspring[chromosome_idx,random_gene_idx]=random_gene
    return offspring

### Function that selects parents based on a tournament
### Random parents are selected, which then fight against one another and the ones with the best fitness are selected
def parent_selection_func(fitness, num_parents, ga_instance):
    parents = np.empty((num_parents, ga_instance.population.shape[1]))
    parents_indices = []
    for parent_num in range(num_parents):
        # Generate random indices for the candiadate solutions.
        rand_indices = np.random.randint(low=0.0, high=len(fitness), size=ga_instance.K_tournament-1)
        K_fitnesses = fitness[rand_indices]
        selected_parent_idx = rand_indices[np.where(K_fitnesses == np.max(K_fitnesses))[0][0]]
        # Append the index of the selected parent.
        parents_indices.append(selected_parent_idx)
        # Insert the selected parent.
        parents[parent_num, :] = ga_instance.population[selected_parent_idx, :].copy()
    return parents,np.array(parents_indices)


### Function that register the elapsed time between the first generation and the current one
### It saves the values in a global variable which is used in other functions
def on_fitness(ga_instance, population_fitness):
    end_time = time.time()
    elapsed_time=end_time-start_time
    global generation_time
    generation_time.append(elapsed_time)
    
    

################# Writing functions for the solutions #########################
    
    
### Function that parses an individual into a coded string
### Returns a string based on three different versions depending on the fitness and combination
def parse_individual(fitness,combination):
    string=""
    ### VERSION WITHOUT FITNESS: course identifier SPACE .... course identifier SKIPLINE
    if fitness is None and combination is not None:
        for course in combination:
            string=string+str(course)+" "
        string=string+'\n'
    ### VERSION WITHOUT COMBINATION: fitness value SPACE soft skill score SPACE ..... soft skill score SKIPLINE
    if combination is None and type(fitness) is list:
        #overall fitness
        string=string+'{0:.3f}'.format(fitness[0])
        #soft skills scores
        for soft_skill_score in fitness[1]:
            string=string+" "+'{0:.3f}'.format(soft_skill_score)
        string=string+'\n'
    ### VERSION WITH COMBINATION AND FITNESS: course identifier SPACE .... course identifier fitness value SKIPLINE
    if combination is not None and fitness is not None:
        for course in combination:
            string=string+str(course)+" "
        string=string+'{0:.3f}'.format(fitness[0])        
        if type(fitness) is tuple:
            #soft skills scores            
            for soft_skill_score in fitness[1]:
                string=string+" "+'{0:.3f}'.format(soft_skill_score)
        string=string+'\n'        
    return string


### Function that writes the fitness run file of the initial and last population
### There are 200 individuals (lines), 100 of the first and 100 of the last generation
### Calls the parse individual function without the combination version
def write_fit_run(pop_init_fitness,pop_last_fitness,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed):
    string_file=""
    for i in range(2):
        if i==0:
            population_fitness=pop_init_fitness
        else:
            population_fitness=pop_last_fitness
        for i in range(len(population_fitness[0])):
            #List containing the overall fitness and the soft skill scores
            individual_fitness=[]
            individual_fitness.append(population_fitness[0][i])
            individual_fitness.append(population_fitness[1][i])
            string_file=string_file+parse_individual(individual_fitness,None)
    #file_name
    file_name=path+"fit_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    file=open(file_name,"w")
    file.write(string_file)
    file.close()
    return string_file


### Function that writes the population run file of the initial and last population
### There are 200 individuals (lines), 100 of the first and 100 of the last generation
### Then, there are 200 lines, where each shows the combination solutions
### Receives as parameter the string file of the fit run file and adds the combinations
### Calls the parse individual function with the combination version 
def write_pop_run(initial_population,last_population,fitness_string_file,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed):
    #sort the solutions of the initial and last population
    initial_population.sort()
    last_population.sort()
    #fitness string file of both populations
    string_file=fitness_string_file
    for i in range(2):
        if i==0:
            population=initial_population
        else:
            population=last_population
        for combination in population:
            string_file=string_file+parse_individual(None,combination)
    #file_name
    file_name=path+"pop_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    file=open(file_name,"w")
    file.write(string_file)
    file.close()

### Function that writes the best solution run file
### It writes the best solution calling the parse individual function
def write_best_sol_run(best_solution,best_fitness,
              student_id,domain_id,score_function,compensatory,number_generations,
              crossover_probability,mutation_probability,seed):
    #sort the best solution
    best_solution.sort()
    #parsing the solution
    string_file=parse_individual(best_fitness, best_solution)
    #file name
    file_name=path+"bestsol_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    file=open(file_name,"w")
    file.write(string_file)
    file.close()
    
### Function that writes the best all run file
### It writes the best solutions per generation along the elapsed time taken
### The number of lines is equal to n+1 generations
def write_best_all_run(best_fitness_by_gen,
              student_id,domain_id,score_function,compensatory,number_generations,
              crossover_probability,mutation_probability,seed):
    global generation_time
    for j in range(len(best_fitness_by_gen)):
        if j==0:
            string_file="0.000 "+'{0:.3f}'.format(best_fitness_by_gen[j])+'\n'
        else:
            string_file=string_file+'{0:.3f}'.format(generation_time[j-1])+" "+'{0:.3f}'.format(best_fitness_by_gen[j])+'\n'
    #file name
    file_name=path+"bestall_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    file=open(file_name,"w")
    file.write(string_file)
    file.close()
    
### Function that calls the other writing functions based on the run results  
def write_files(solution,solution_fitness,best_fitness_by_gen,initial_population,last_population,pop_init_fitness,pop_last_fitness,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed):
    fitness_string_file=write_fit_run(pop_init_fitness,pop_last_fitness,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed)
    
    write_pop_run(initial_population,last_population,fitness_string_file,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed)
    write_best_sol_run(solution,solution_fitness,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed)
    write_best_all_run(best_fitness_by_gen,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed)
    

    
###################### Parameter input for the script #########################

### Formatting of the parser
parser = ArgumentParser(description="Genetic Algorithm Script",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--student_row", default=0,type=int, help="Student id row")
parser.add_argument("-d", "--domain_id", default=1, type=int, help="Domain id EE=1,IS=2,MX=3,NU=4")
parser.add_argument("-f", "--score_function", default=1, type=int, help="Score function Linear=1,Logistic=2,Quadratic=3")
parser.add_argument("-c", "--compensatory", default=False, type=lambda x: (str(x).lower() == 'true' or str(x).lower() == 't' or str(x)=='1'), help="Compensatory=True, Partially Compensatory=False")
parser.add_argument("-g", "--number_generations", default=10, type=int, help="Number of generations")
parser.add_argument("-x", "--crossover_probability", default=0.65, type=float, help="Probability of crossover")
parser.add_argument("-m", "--mutation_probability", default=0.15, type=float, help="Probability of mutation")
parser.add_argument("-n", "--sol_per_pop", default=100, type=int, help="Population size")
parser.add_argument("-e", "--keep_elitism", default=1, type=int, help="Number of combinations kept after each generation")
parser.add_argument("-i", "--seed", default=1, type=int, help="Seed")
### Parsing the arguments
args = vars(parser.parse_args())

### Parameter reading
student_row=args["student_row"]
domain_id=args["domain_id"]
score_function=args["score_function"]
compensatory=args["compensatory"]
number_generations=args["number_generations"]
crossover_probability=args["crossover_probability"]
mutation_probability=args["mutation_probability"]
sol_per_pop=args["sol_per_pop"]
keep_elitism=args["keep_elitism"]
seed=args["seed"]

### Setting the Genetic Algorithm parameters

#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
#Considering only students with more than 5 courses
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
#Shrinking the dataset for domain of interest
real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==domain_id]
real_data_stage2=real_data_stage2.reset_index(drop=True)
#student id
student_id=real_data_stage2.iloc[student_row,0]
#Number of followed courses
N_courses_followed=real_data_stage2.iloc[student_row,25]
#Thresholds
thresholds=rdf.get_thresholds()
#Course effects
courses_effects=rdf.get_courses_effects()
#Minimum soft skill score
min_skill=1
#Maximum soft skill score
max_skill=4       
#Student random effects 
theta=rdf.get_student_random_effect(student_id)
#Expected profile of domain of interest
desired_outcome=rdf.get_desired_outcome(domain_id)
#List of possible courses for the gene space
possible_courses=rdf.get_courses_domain(domain_id)      
#Number of parent combinations mating  
num_parents_mating=int(sol_per_pop/2)
#Number of parents to keep for the next generation
keep_parents=keep_elitism
#Parent selection function
parent_selection_type=parent_selection_func
#Time variables
start_time = time.time()
generation_time=[]


############################ Genetic Algorithm Launch #########################



ga_instance = pygad.GA(num_generations=number_generations,
                   num_parents_mating=num_parents_mating,
                   fitness_func=fitness_func,
                   crossover_type=unifcrossover_func,
                   mutation_type=mutation_func,
                   crossover_probability=crossover_probability,
                   mutation_probability=mutation_probability,
                   #fitness_batch_size=10,
                   sol_per_pop=sol_per_pop,
                   num_genes=int(N_courses_followed),
                   allow_duplicate_genes=False,
                   parent_selection_type=parent_selection_type,
                   gene_type=int,gene_space=possible_courses,
                   keep_elitism=keep_elitism,
                   on_fitness=on_fitness,
                   random_seed=seed,
                   suppress_warnings=True
                   )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution.sort()
print("Parameters of the best solution :",solution)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Crossover probability:",ga_instance.crossover_probability)
print("Mutation probability:",ga_instance.mutation_probability)
print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

#initial population
initial_population=ga_instance.initial_population.copy()
#last population
last_population=ga_instance.population.copy()
#fitness of initial population
pop_init_fitness=fitness_func_batch(None,initial_population,None)
#fitness of last population
pop_last_fitness=fitness_func_batch(None,last_population,None)
#best fitness by generation
best_fitness_by_gen=ga_instance.best_solutions_fitness
#Fitness with scores of the solution
solution_fitness=fitness_func(None,solution,None)
#directory setting
file_directory=rdf.toStringfilestructure(domain_id,compensatory,score_function,None,folder="/tests/")
path="/home/luis.pinos-ullauri"+file_directory
mode = 0o777    
if not os.path.exists(path):
    os.makedirs(path,mode)
#writing results
write_files(solution,solution_fitness,best_fitness_by_gen,initial_population,last_population,pop_init_fitness,pop_last_fitness,
                  student_id,domain_id,score_function,compensatory,number_generations,
                  crossover_probability,mutation_probability,seed)
