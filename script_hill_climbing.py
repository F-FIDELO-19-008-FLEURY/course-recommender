# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:41:54 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
from scipy.special import expit
import numpy as np
import time
import os
import random as rand
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

def generate_combination(possible_courses,N_courses_followed):
    combination=[]
    while len(combination)<N_courses_followed:
        random_course=np.random.choice(possible_courses)
        if random_course not in combination:
            combination.append(random_course)
    return combination

### Function that register the elapsed time between each 1000 checks of the hill climbing
def time_register():
    end_time = time.time()
    elapsed_time=end_time-start_time
    global generation_time
    generation_time.append(elapsed_time)
    

def hill_climbing(objective_function, generate_combination, stopping_criterion,possible_courses,N_courses_followed):
    current_solution=generate_combination(possible_courses,N_courses_followed)
    cont=0
    flag=stopping_criterion
    solutions.append(current_solution)
    while flag>0:        
        current_fitness=objective_function(1,current_solution,0)
        potential_solution=generate_combination(possible_courses,N_courses_followed)
        potential_fitness=objective_function(1,potential_solution,0)
        if potential_fitness>current_fitness:
            current_solution=potential_solution
            flag=stopping_criterion
        else:
            flag=flag-1
        cont+=1
        if cont%100==0:
            time_register()
            solutions.append(current_solution)
    return current_solution

### Function that writes the best all run file
### It writes the best solutions per generation along the elapsed time taken
### The number of lines is equal to n+1 generations
def write_best_all_run(path,student_id,domain_id,score_function,compensatory,seed):
    global generation_time
    global solutions
    for j in range(len(solutions)):
        if j==0:
            string_file="0.000 "+'{0:.3f}'.format(solutions[j])+'\n'
        else:
            string_file=string_file+'{0:.3f}'.format(generation_time[j-1])+" "+'{0:.3f}'.format(solutions[j])+'\n'
    
    #file title
    file_name=path+"hill_climbing_"+str(student_id)+"_"+str(seed)+".txt"
    file=open(file_name,"w")
    file.write(string_file)
    file.close()


###################### Parameter input for the script #########################

### Formatting of the parser
parser = ArgumentParser(description="Hill Climbing Script",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--student_row", default=8,type=int, help="Student id row")
parser.add_argument("-d", "--domain_id", default=4, type=int, help="Domain id EE=1,IS=2,MX=3,NU=4")
parser.add_argument("-f", "--score_function", default=1, type=int, help="Score function Linear=1,Logistic=2,Quadratic=3")
parser.add_argument("-c", "--compensatory", default=True, type=lambda x: (str(x).lower() == 'true' or str(x).lower() == 't' or str(x)=='1'), help="Compensatory=True, Partially Compensatory=False")
parser.add_argument("-n", "--stop criterion", default=100, type=int, help="Number of generations")
parser.add_argument("-i", "--seed", default=1, type=int, help="Seed")
### Parsing the arguments
args = vars(parser.parse_args())

### Parameter reading
student_row=args["student_row"]
domain_id=args["domain_id"]
score_function=args["score_function"]
compensatory=args["compensatory"]
stop_criterion=args["stop criterion"]
seed=args["seed"]


#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
#Domain 1: EE
real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==domain_id]
real_data_stage2=real_data_stage2.reset_index(drop=True)
#Thresholds
thresholds=rdf.get_thresholds()
#Course Effects
courses_effects=rdf.get_courses_effects()
#student id
student_id=real_data_stage2.iloc[student_row,0]
#Number of followed courses
N_courses_followed=real_data_stage2.iloc[student_row,25]
min_skill=1#Minimum Soft skill proficiency
max_skill=4#Maximum Soft skill proficiency
#Student Effect
theta=rdf.get_student_random_effect(student_id)
#Desired outcome
desired_outcome=rdf.get_desired_outcome(domain_id)
#get possible courses
possible_courses=rdf.get_courses_domain(domain_id)
rand.seed(seed)
#Time variables
start_time = time.time()
generation_time=[]
solutions=[]
file_directory=rdf.toStringfilestructure(domain_id,compensatory,score_function,student_id,folder="/results/")
path="/home/luis.pinos-ullauri"+file_directory
mode = 0o777    
if not os.path.exists(path):
    os.makedirs(path,mode)

sol=hill_climbing(fitness_func,generate_combination,stop_criterion,possible_courses,N_courses_followed)
sol.sort()
print(sol)
print(fitness_func(None,sol,0))
write_best_all_run(student_id,domain_id,score_function,compensatory,seed)






