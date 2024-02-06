# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:36:37 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
import itertools
import pandas as pd
import numpy as np
from scipy.special import expit
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

######################## User Defined Functions ###############################

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
        if solution[i]!=0 and solution[i]!=104 and solution[i]!=105:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]]
        elif solution[i]==104 or solution[i]==105:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]-1]
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




################## End of  User Defined Functions #############################


### NEED FURTHER THOUGHTS INTO WHICH PLOT TO USE OR MAYBE JUST A TABLE


###################### Parameter input for the script #########################

### Formatting of the parser
parser = ArgumentParser(description="Genetic Algorithm Script",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--student_row", default=0,type=int, help="Student id row")
parser.add_argument("-d", "--domain_id", default=4, type=int, help="Domain id EE=1,IS=2,MX=3,NU=4")
parser.add_argument("-f", "--score_function", default=1, type=int, help="Score function Linear=1,Logistic=2,Quadratic=3")
parser.add_argument("-c", "--compensatory", default=False, type=lambda x: (str(x).lower() == 'true' or str(x).lower() == 't' or str(x)=='1'), help="Compensatory=True, Partially Compensatory=False")
parser.add_argument("-g", "--number_generations", default=100, type=int, help="Number of generations")
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
#Removing the one student with 12 courses
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
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


################## Brute Force Estimation Computing Time ######################




#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
#Domain 1: EE
real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==4]
real_data_stage2=real_data_stage2.reset_index(drop=True)
#Thresholds
thresholds=rdf.get_thresholds()
#Course Effects
courses_effects=rdf.get_courses_effects()
student_id=real_data_stage2.iloc[0,0]
domain_id=real_data_stage2.iloc[0,26]
N_courses_followed=real_data_stage2.iloc[0,25]
min_skill=1#Minimum Soft skill proficiency
max_skill=4#Maximum Soft skill proficiency
#Compensatory boolean variable
compensatory=True
#Score function flag variable
score_function=2#Linear
#score_func=2#Logistic
#Student Effect
theta=rdf.get_student_random_effect(student_id)
#Desired outcome
desired_outcome=rdf.get_desired_outcome(domain_id)
#get possible courses
possible_courses=rdf.get_courses_domain(domain_id)
N_possible_combinations=math.comb(len(possible_courses),N_courses_followed)
fitness_spread=pd.DataFrame(np.zeros(shape=(N_possible_combinations,14)))
fitness_spread.columns=['combination_id','student_id','domain_id','gen_fitness','scoress1','scoress2',
                        'scoress3','scoress4','scoress5','scoress6','scoress7',
                        'scoress8','scoress9','scoress10']
"""
i=0
for solution in itertools.combinations(possible_courses, N_courses_followed):
    current_fitness=fitness_func(None,solution,0)
    fitness_spread.iloc[i,0]=i    
    fitness_spread.iloc[i,1]=student_id
    fitness_spread.iloc[i,2]=domain_id
    fitness_spread.iloc[i,3]=current_fitness[0]
    fitness_spread.iloc[i,4]=current_fitness[1][0]
    fitness_spread.iloc[i,5]=current_fitness[1][1]
    fitness_spread.iloc[i,6]=current_fitness[1][2]
    fitness_spread.iloc[i,7]=current_fitness[1][3]
    fitness_spread.iloc[i,8]=current_fitness[1][4]
    fitness_spread.iloc[i,9]=current_fitness[1][5]
    fitness_spread.iloc[i,10]=current_fitness[1][6]
    fitness_spread.iloc[i,11]=current_fitness[1][7]
    fitness_spread.iloc[i,12]=current_fitness[1][8]
    fitness_spread.iloc[i,13]=current_fitness[1][9]
    i+=1
if compensatory and score_function==1:
    fitness_spread.to_csv("./real_data/fitness_spread_comp_linear.csv")
if not compensatory and score_function==1:
    fitness_spread.to_csv("./real_data/fitness_spread_parcomp_linear.csv")
if compensatory and score_function==2:
    fitness_spread.to_csv("./real_data/fitness_spread_comp_logistic.csv")
if not compensatory and score_function==2:
    fitness_spread.to_csv("./real_data/fitness_spread_parcomp_logistic.csv")
if compensatory and score_function==3:
    fitness_spread.to_csv("./real_data/fitness_spread_comp_quadratic.csv")
if not compensatory and score_function==3:
    fitness_spread.to_csv("./real_data/fitness_spread_parcomp_quadratic.csv")

"""
################# End of Brute Force Estimation Computing Time ################