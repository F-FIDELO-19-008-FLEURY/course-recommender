# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:47:42 2023

@author: luis.pinos-ullauri
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:36:37 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
import pandas as pd
import numpy as np
from scipy.special import expit

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
        if solution[i]!=0 and solution[i]!=104 and solution[i]!=105 and solution[i]!=106:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]]
        elif solution[i]==104 or solution[i]==105:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]+courses_effects.iloc[soft_skill_id,solution[i]-1]
        elif solution[i]==103 or solution[i]==106:
            linear_combination=linear_combination+courses_effects.iloc[soft_skill_id,0]
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

################## Data Analysis ######################


#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
#real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==1]
min_skill=1#Minimum Soft skill proficiency
max_skill=4#Maximum Soft skill proficiency
#Compensatory boolean variable
comp=False
#Score function flag variable
score_func=2#Linear
#Read recommendations
recommendations=rdf.read_recommendations(score_func,comp)
real_data_stage2=real_data_stage2.reset_index(drop=True)
#Thresholds
thresholds=rdf.get_thresholds()
#Course Effects
courses_effects=rdf.get_courses_effects()
comparison=pd.DataFrame(np.zeros(shape=(len(real_data_stage2),5)))
for i in range(len(real_data_stage2)):
    student_id=real_data_stage2.iloc[i,0]
    domain_id=real_data_stage2.iloc[i,26]
    N_courses_followed=real_data_stage2.iloc[i,25]
    #Student Effect
    theta=rdf.get_student_random_effect(student_id)
    #Desired outcome
    desired_outcome=rdf.get_desired_standard(domain_id)
    #get possible courses
    possible_courses=rdf.get_courses_domain(domain_id)
    actual_courses=real_data_stage2.iloc[i,13:24].values.flatten().tolist()
    actual_courses=actual_courses[0:N_courses_followed]
    for j in range(len(actual_courses)):
        actual_courses[j]=int(actual_courses[j]) 
    #send that list as 2nd parameter
    real_fitness=fitness_func(None,actual_courses,0)
    recommendation_fitness=recommendations.iloc[i,11]
    comparison.iloc[i,0]=student_id
    comparison.iloc[i,1]=domain_id
    comparison.iloc[i,2]=recommendation_fitness
    comparison.iloc[i,3]=real_fitness
    comparison.iloc[i,4]=(recommendation_fitness-real_fitness)/abs(real_fitness)
    
comparison.columns=['student_id','domain_id','recommendation_fitness','real_fitness','increase']

if score_func==1 and comp:
    comparison.to_csv("./real_data/comparison_lin_comp.csv")
if score_func==1 and not comp:
    comparison.to_csv("./real_data/comparison_lin_parcomp.csv")
if score_func==2 and comp:
    comparison.to_csv("./real_data/comparison_quad_comp.csv")
if score_func==2 and not comp:
    comparison.to_csv("./real_data/comparison_quad_parcomp.csv")
if score_func==3 and comp:
    comparison.to_csv("./real_data/comparison_exp_comp.csv")
if score_func==3 and not comp:
    comparison.to_csv("./real_data/comparison_exp_parcomp.csv")







################# End of Brute Force Estimation Computing Time ################