# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:36:37 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
from scipy.special import expit
import itertools
import time
import pandas as pd
import numpy as np

######################## User Defined Functions ###############################

### Function that returns the overall fitness of the solution across all soft skill dimensions
### It checks whether the fitness amongst the different dimensions is compensatoryensated or not
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
def soft_skill_estimation_mean(thresholds,courses_effects,theta,solution,soft_skill_id):
    linear_combination=0
    for i in range(len(solution)):
        if solution[i]!=0:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]]
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



################## End of  User Defined Functions #############################


### NEED FURTHER THOUGHTS INTO WHICH PLOT TO USE OR MAYBE JUST A TABLE


################## Brute Force Estimation compensatoryuting Time ######################


#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
#Domain 1: NU
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
compensatory=True
#Score function flag variable
score_function=1#Linear
#score_function=2#Logistic
#Student Effect
theta=rdf.get_student_random_effect(student_id)
#Desired outcome
desired_outcome=rdf.get_desired_standard(domain_id)
#get possible courses
possible_courses=rdf.get_courses_domain(domain_id)
calculation_time=pd.DataFrame(np.zeros(shape=(1,15)))
calculation_time.columns=['combination_index','c1','c2','c3','c4',
                          'c5','c6','c7','c8',
                          'c9','c10','c11','fitness','time(s)','Ncombs']
best_fitness=-1
best_solution=[]
start_time = time.time()
i=0
for solution in itertools.combinations(possible_courses, N_courses_followed):
    current_fitness=fitness_func(None,solution,0)
    if current_fitness>best_fitness:
        best_fitness=current_fitness
        best_solution=solution    
    if i==0 or i%2000==0:        
        end_time = time.time()
        best_solution=list(best_solution)
        best_solution.sort()
        elapsed_time = end_time - start_time
        combs=math.comb(len(possible_courses),len(solution))
        calculation_time.loc[i,:]=[i,best_solution[0],best_solution[1],
                                   best_solution[2],best_solution[3],best_solution[4],
                                   best_solution[5],best_solution[6],best_solution[7],
                                   best_solution[8],best_solution[9],0,
                                   best_fitness,elapsed_time,combs]        
        calculation_time.to_csv("./real_data/combinations_cal_time_bf_NU.csv")
    i+=1


################# End of Brute Force Estimation compensatoryuting Time ################