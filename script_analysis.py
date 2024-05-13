# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:05:26 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

############### User-defined functions for the Genetic Algorithm ##############

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

### Function that checks which dimensions are bellow the expected profile
### If they are bellow, it appends a 0, if not, it appends a +1
### It returns a list of binary values
def soft_skill_dimension_comparison(average_dimensions,score_function):
    dimensions_passed=[]
    for i in range(len(average_dimensions)):#soft skill id from 0 to 9
        if score_function==1:
            score_threshold=linear(desired_outcome[i],i)
        if score_function==2:
            score_threshold=logistic(desired_outcome[i],i)
        if score_function==3:
            score_threshold=quadratic(desired_outcome[i],i)
        if average_dimensions[i]-score_threshold>=0:
            dimensions_passed.append(1)
        else:
            dimensions_passed.append(0)
    return dimensions_passed



################# Analysis functions for the solutions ########################
  

### Function that writes the comparison between the recommendation fitness and the actual fitness
def write_comparison_avg_fitness(domain_id,score_function,compensatory,comparison_df,seed_init,seed_end,path):
    global theta
    #overall loop of all students for an specific domain
    for i in range(len(real_data_stage2)):
        #student id
        student_id=int(real_data_stage2.iloc[i,0])       
        #Student random effects 
        theta=rdf.get_student_random_effect(student_id)
        #Number of followed courses
        N_courses_followed=real_data_stage2.iloc[i,25]
        actual_courses=real_data_stage2.iloc[i,13:24].values.flatten().tolist()
        actual_courses=actual_courses[0:N_courses_followed]
        for j in range(len(actual_courses)):
            actual_courses[j]=int(actual_courses[j])
        #send that list as 2nd parameter
        real_fitness=fitness_func(None,actual_courses,None)
        average_recommendation_fitness=rdf.read_average_solution(student_id, domain_id, score_function, compensatory, number_generations, crossover_probability, mutation_probability, seed_init, seed_end,True)
        dimensions_recommended=soft_skill_dimension_comparison(average_recommendation_fitness[1:len(average_recommendation_fitness)],score_function)
        dimensions_real=soft_skill_dimension_comparison(real_fitness[1],score_function)
        comparison_df.iloc[i,0]=int(student_id)
        comparison_df.iloc[i,1]=domain_id
        comparison_df.iloc[i,2]=compensatory
        comparison_df.iloc[i,3]=score_function
        comparison_df.iloc[i,4]=average_recommendation_fitness[0]
        comparison_df.iloc[i,5]=real_fitness[0]
        comparison_df.iloc[i,6]=(average_recommendation_fitness[0]-real_fitness[0])/real_fitness[0]
        for k in range(1,len(average_recommendation_fitness)):
            comparison_df.iloc[i,6+k]=(average_recommendation_fitness[k]-real_fitness[1][k-1])/real_fitness[1][k-1]
        for l in range(len(dimensions_recommended)):
            comparison_df.iloc[i,17+l]=dimensions_recommended[l]-dimensions_real[l]
        comparison_df.iloc[i,27]=sum(dimensions_recommended)
        comparison_df.iloc[i,28]=sum(dimensions_real)
        comparison_df.iloc[i,29]=N_courses_followed    

    #column names for the dataframe
    comparison_df.columns=['student_id','domain_id','compensatory','score_function','recommendation_fitness',
                        'real_fitness','increase',
                        'increase_ssk1','increase_ssk2','increase_ssk3','increase_ssk4','increase_ssk5',
                        'increase_ssk6','increase_ssk7','increase_ssk8','increase_ssk9','increase_ssk10',
                        'dim_gain_ssk1','ddim_gain_ssk2','dim_gain_ssk3','dim_gain_ssk4','dim_gain_ssk5',
                        'dim_gain_ssk6','dim_gain_ssk7','dim_gain_ssk8','dim_gain_ssk9','dim_gain_ssk10',                                      
                        'dimensions_passed_recommended','dimensions_passed_real','courses_followed']
    #writing the result files
    comparison_df.to_csv(path+"comparison_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed_init)+"_"+str(seed_end)+".csv")


### Function that reads the comparison files between the recommendation fitness and the actual fitness
def read_comparison_avg_fitness(domain_id,score_function,compensatory,seed_init,seed_end):
    #directory setting
    file_directory=rdf.toStringfilestructure(domain_id,compensatory,folder="./analysis/")
    dataset_domain_comp_function=[]
    if score_function==1:
        score_function_as_string="linear_"
    elif score_function==2:
        score_function_as_string="logistic_"
    elif score_function==3:
        score_function_as_string="quadratic_"
    path_to_read=file_directory+score_function_as_string+"comparison_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed_init)+"_"+str(seed_end)+".csv"
    if os.path.exists(path_to_read):        
        dataset_domain_comp_function=pd.read_csv(path_to_read)
        dataset_domain_comp_function.drop(dataset_domain_comp_function.columns[0],axis=1,inplace=True)
    return dataset_domain_comp_function

def read_loop(domain_id,seed_init,seed_end):
    for i in range(2):
        if i==0:
            compensatory=True
        else:
            compensatory=False
        for j in range(3):
            if i==0 and j==0:
                domain_dataset=read_comparison_avg_fitness(domain_id,j+1,compensatory,seed_init,seed_end)
            else:
                domain_dataset=pd.concat([domain_dataset,read_comparison_avg_fitness(domain_id,j+1,compensatory,seed_init,seed_end)])
    return domain_dataset

### Function that shows the boxplot
def boxplot(domain_id,score_function,compensatory,comparison_df,seed_init,seed_end):
    domain_dataset=read_loop(domain_id,seed_init,seed_end)
    title=""
    fs = 10  # fontsize
    compensatory_increase=pd.concat([domain_dataset.loc[(domain_dataset["compensatory"]==True) & (domain_dataset["score_function"]==1)]["increase"],
                                     domain_dataset.loc[(domain_dataset["compensatory"]==True) & (domain_dataset["score_function"]==2)]["increase"],
                                     domain_dataset.loc[(domain_dataset["compensatory"]==True) & (domain_dataset["score_function"]==3)]["increase"]         
                                     ],axis=1)
    parcompensatory_increase=pd.concat([domain_dataset.loc[(domain_dataset["compensatory"]==False) & (domain_dataset["score_function"]==1)]["increase"],
                                     domain_dataset.loc[(domain_dataset["compensatory"]==False) & (domain_dataset["score_function"]==2)]["increase"],
                                     domain_dataset.loc[(domain_dataset["compensatory"]==False) & (domain_dataset["score_function"]==3)]["increase"]         
                                     ],axis=1)
    labels=["Linear","Logistic","Quadratic"]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,7),dpi=300)
    if domain_id==1:
        domain_str="Energy and Environment"
    elif domain_id==2:
        domain_str="Industry and Services"
    elif domain_id==3:
        domain_str="Ecomaterials and Civil Engineering"
    elif domain_id==4:
        domain_str="Digital"
    title=title+domain_str
    axs[0].boxplot(compensatory_increase,labels=labels, showmeans=True)
    axs[0].set_title(title+" Compensatory", fontsize=fs)
    axs[1].boxplot(parcompensatory_increase,labels=labels, showmeans=True)
    axs[1].set_title(title+" Partially Compensatory", fontsize=fs)
   
    for ax in axs.flat:
        ax.yaxis.grid(True)
    
    fig.subplots_adjust(hspace=0.4)
    plt.show()
    return domain_dataset
    


###################### Parameter input for the script #########################

### Formatting of the parser
parser = ArgumentParser(description="Reading Recommendations Script",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--domain_id", default=2, type=int, help="Domain id EE=1,IS=2,MX=3,NU=4")
parser.add_argument("-f", "--score_function", default=1, type=int, help="Score function Linear=1,Logistic=2,Quadratic=3")
parser.add_argument("-c", "--compensatory", default=False, type=lambda x: (str(x).lower() == 'true' or str(x).lower() == 't' or str(x)=='1'), help="Compensatory=True, Partially Compensatory=False")
parser.add_argument("-g", "--number_generations", default=100, type=int, help="Number of generations")
parser.add_argument("-x", "--crossover_probability", default=0.80, type=float, help="Probability of crossover")
parser.add_argument("-m", "--mutation_probability", default=0.25, type=float, help="Probability of mutation")
parser.add_argument("-a", "--action", default=1, type=int, help="Actions: write=1,boxplot=2")
parser.add_argument("-i", "--seed_init", default=1, type=int, help="Seed")
parser.add_argument("-j", "--seed_end", default=100, type=int, help="Seed")
### Parsing the arguments
args = vars(parser.parse_args())

### Parameter reading
domain_id=args["domain_id"]
score_function=args["score_function"]
compensatory=args["compensatory"]
number_generations=args["number_generations"]
crossover_probability=args["crossover_probability"]
mutation_probability=args["mutation_probability"]
action=args["action"]
seed_init=args["seed_init"]
seed_end=args["seed_end"]


#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
#Considering only students with more than 5 courses
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
#Shrinking the dataset for domain of interest
real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==domain_id]
real_data_stage2=real_data_stage2.reset_index(drop=True)
#Thresholds
thresholds=rdf.get_thresholds()
#Course effects
courses_effects=rdf.get_courses_effects()
#Student random effects
theta=[]
#Minimum soft skill score
min_skill=1
#Maximum soft skill score
max_skill=4
#Expected profile of domain of interest
desired_outcome=rdf.get_desired_outcome(domain_id)
#List of possible courses for the gene space
possible_courses=rdf.get_courses_domain(domain_id) 
#comparison dataframe to be filled in the following loop
comparison=pd.DataFrame(np.zeros(shape=(len(real_data_stage2),30)))
#directory setting
file_directory=rdf.toStringfilestructure(domain_id,compensatory,score_function,None,folder="/analysis/")
path="/home/luis.pinos-ullauri"+file_directory
mode = 0o777    
if not os.path.exists(path):
    os.makedirs(path,mode)
if action==1:
    write_comparison_avg_fitness(domain_id, score_function, compensatory, comparison, seed_init, seed_end,path)
if action==2:
    y=boxplot(domain_id, score_function, compensatory, comparison, seed_init, seed_end)




