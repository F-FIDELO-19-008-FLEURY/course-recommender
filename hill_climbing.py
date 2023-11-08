# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:41:54 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
from scipy.special import expit
import numpy as np

######################## User Defined Functions ###############################

### Function that returns the overall fitness of the solution across all soft skill dimensions
### It checks whether the fitness amongst the different dimensions is compensated or not
### as well as which scoring function to use
def fitness_func(ga_instance, solution, solution_idx):
    soft_skill_scores=[]
    if comp:
        fitness=0
    else:
        fitness=1
    for i in range(10):#soft skill id from 0 to 9
        estimated_outcome=soft_skill_estimation_mean(thresholds, courses_effects, theta[i], solution,i)
        if score_func==1:
            score=linear(estimated_outcome,desired_outcome[i])
        elif score_func==2:
            score=logistic(estimated_outcome,desired_outcome[i])
        elif score_func==3:
            score=quadratic(estimated_outcome,desired_outcome[i])
        if ga_instance is None:
            soft_skill_scores.append(score)            
        if comp:
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
def linear(estimated_skill,goal_skill):
    #function min value
    f_min=min_skill-goal_skill
    #function max value
    f_max=max_skill-goal_skill    
    return (estimated_skill-goal_skill-f_min)/(f_max-f_min)
   
### Logistic scoring function
### Stricter scoring with penalisations and bonuses if the
### soft skill proficiency mean reaches or not the expcted profile
### Scoring function rescaled so that it ranges from 0 to 1     
def logistic(estimated_skill,goal_skill):
    #crossing value with linear function at estimated_goal=goal_skill
    fcrossing=(goal_skill-min_skill)/(max_skill-min_skill)
    return (1)/(1+((1-fcrossing)/fcrossing)*pow(math.e,3*(goal_skill-estimated_skill)))

### Quadratic Root scoring function
### Less demanding scoring that allows an easier scoring if the
### soft skill proficiency mean reaches or not the expcted profile
### Scoring function rescaled so that it ranges from 0 to 1  
def quadratic(estimated_skill,goal_skill):
    #crossing value with linear function at estimated_goal=goal_skill
    fcrossing=(goal_skill-min_skill)/(max_skill-min_skill)
    #function min value
    f_min=min_skill-goal_skill
    #function max value
    f_max=max_skill-goal_skill
    if estimated_skill<=goal_skill:            
        return -1*pow(estimated_skill-goal_skill,2)/(-f_min)*fcrossing+fcrossing
    else:
        return 1*pow(estimated_skill-goal_skill,2)/(f_max)*(1-fcrossing)+fcrossing

### Function that estimates the soft skill mean based on the ordinal logistic regression model
### It calculates the probability of each level and estimates the mean SUM x*P(X=x)
def soft_skill_estimation_mean(thresholds,courses_effects,theta,solution,soft_skill_id):
    linear_combination=0
    for i in range(len(solution)):
        if solution[i]!=0 and solution[i]!=104 and solution[i]!=105:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]]
        elif solution[i]==104 or solution[i]==105:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]-1]
    linear_combination=linear_combination+theta
    eta12=thresholds.iloc[soft_skill_id,0]-linear_combination
    eta23=thresholds.iloc[soft_skill_id,1]-linear_combination
    eta34=thresholds.iloc[soft_skill_id,2]-linear_combination
    p_1=expit(eta12)
    p_2=expit(eta23)-p_1
    p_3=expit(eta34)-p_1-p_2
    p_4=1-p_3-p_2-p_1
    expected_outcome=1*p_1+2*p_2+3*p_3+4*p_4
    return expected_outcome

def generate_initial_combination(possible_courses,N_courses_followed):
    combination=[]
    while len(combination)<N_courses_followed:
        random_course=np.random.choice(possible_courses)
        if random_course not in combination:
            combination.append(random_course)
    return combination

def generate_neighbor(combination,possible_courses,N_courses_followed):
    if combination is None:
        current_solution=generate_initial_combination(possible_courses,N_courses_followed)
        return current_solution
    flag=False
    count=0
    random_course=np.random.choice(possible_courses)
    random_course_idx = np.random.choice(N_courses_followed)    
    while flag!=True and count<=10:
        combination_as_list=list(combination)
        if random_course in combination_as_list:#random gene is already in the current chromosome
            random_course=np.random.choice(possible_courses)
            count+=1
        else:
            #print(combination)
            current_solution=combination.copy()
            current_solution[random_course_idx]=random_course
            flag=True       
    return current_solution
    

def hill_climbing(objective_function, generate_neighbor, stopping_criterion,possible_courses,N_courses_followed):
    current_solution=generate_neighbor(None,possible_courses,N_courses_followed)
    cont=0
    flag=stopping_criterion
    #print(current_solution)
    while flag>0:        
        current_fitness=objective_function(None,current_solution,0)
        potential_solution=generate_neighbor(current_solution,possible_courses,N_courses_followed)
        potential_fitness=objective_function(None,potential_solution,0)
        if potential_fitness>current_fitness:
            current_solution=potential_solution
            flag=stopping_criterion
        else:
            flag=flag-1
        cont+=1
        #print(cont)
    print(cont)
    return current_solution


################## End of  User Defined Functions #############################


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
student_id=real_data_stage2.iloc[8,0]
domain_id=real_data_stage2.iloc[8,26]
N_courses_followed=real_data_stage2.iloc[8,25]
min_skill=1#Minimum Soft skill proficiency
max_skill=4#Maximum Soft skill proficiency
#Compensatory boolean variable
comp=True
#Score function flag variable
score_func=1#Linear
#score_func=2#Logistic
#Student Effect
theta=rdf.get_student_random_effect(student_id)
#Desired outcome
desired_outcome=rdf.get_desired_standard(domain_id)
#get possible courses
possible_courses=rdf.get_courses_domain(domain_id)

sol=hill_climbing(fitness_func,generate_neighbor,100,possible_courses,N_courses_followed)
sol.sort()
print(sol)
print(fitness_func(None,sol,0))






