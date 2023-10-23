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
### It checks whether the fitness amongst the different dimensions is compensated or not
### as well as which scoring function to use
def fitness_func(ga_instance, solution, solution_idx):
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
            
        if comp:
            fitness=fitness+score*(1/10)
        else:
            fitness=fitness*score
    return fitness

### Linear scoring function
def linear(estimated_skill,goal_skill):
    #function min value
    f_min=min_skill-goal_skill
    #function max value
    f_max=max_skill-goal_skill    
    return (estimated_skill-goal_skill-f_min)/(f_max-f_min)
   
### Logistic scoring function    
def logistic(estimated_skill,goal_skill):
    #crossing value with linear function at estimated_goal=goal_skill
    fcrossing=(goal_skill-min_skill)/(max_skill-min_skill)
    return (1)/(1+((1-fcrossing)/fcrossing)*pow(math.e,3*(goal_skill-estimated_skill)))

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

### Function that saves a combination of courses into the appropiate cells at the dataframe
def save_recommendations(recommendations,row,combination,N_courses_followed):
    for i in range(N_courses_followed):
        recommendations.iloc[row,i+1]=combination[i]



################## End of  User Defined Functions #############################


### NEED FURTHER THOUGHTS INTO WHICH PLOT TO USE OR MAYBE JUST A TABLE


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


"""
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
calculation_time=pd.DataFrame(np.zeros(shape=(100,4)))
calculation_time.columns=['trial','fitness','time(s)','Ncombs']
for i in range(100):
    calculation_time.iloc[i,0]=i
    best_fitness=-1
    best_solution=[]
    start_time = time.time()
    for solution in itertools.combinations(possible_courses, N_courses_followed):
        current_fitness=fitness_func(None,solution,0)
        if current_fitness>best_fitness:
            best_fitness=current_fitness
            best_solution=solution
    end_time = time.time()
    elapsed_time = end_time - start_time
    combs=math.comb(len(possible_courses),len(solution))
    calculation_time.iloc[i,1]=best_fitness
    calculation_time.iloc[i,2]=elapsed_time
    calculation_time.iloc[i,3]=combs
    print(best_solution)    
    #print(f"Elapsed time: {elapsed_time} seconds")    
    #print(combs," Combinations")
calculation_time.to_csv("./real_data/combinations_cal_time_bf.csv")
"""


################# End of Brute Force Estimation Computing Time ################