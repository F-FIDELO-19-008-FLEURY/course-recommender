# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:59:01 2023

@author: luis.pinos-ullauri
"""


import os.path
import pandas as pd


################## Reading Functions for the real data set ####################


### Function that returns the real data set
### Returns a longitudinal dataframe where each row belongs to a student in a particular stage
### The courses c1 to c11 describe the followed course identifiers up until that stage
def read_real_data():
    #If the file exists, read it and return it
    if os.path.exists("./real_data/full_dataset_coursesRE.csv"):
        real_data=pd.read_csv("./real_data/full_dataset_coursesRE.csv",encoding=('latin1'))
        real_data.drop([real_data.columns[0],real_data.columns[3],
                              real_data.columns[4],real_data.columns[5],
                              real_data.columns[6],real_data.columns[7],
                              real_data.columns[18],real_data.columns[19],
                              real_data.columns[20],real_data.columns[21],
                              real_data.columns[23],real_data.columns[24],
                              real_data.columns[25]],axis=1,inplace=True)
        return real_data
    return []


### Function that returns the average fitness from the different seeds of the same student
def read_average_solution(student_row,domain_id,score_function,compensatory,number_generations,crossover_prob,mutation_prob,seed_init,seed_end,dimensions):  
    average_fitness=0
    average_dimensions=[]
    for seed in range(seed_init,seed_end+1):
        if dimensions is False:
            average_fitness=average_fitness+read_solution(student_row,domain_id,score_function,compensatory,number_generations,crossover_prob,mutation_prob,seed,dimensions)
        else:
            average_dimensions=[a+b for a,b in zip(average_dimensions,read_solution(student_row,domain_id,score_function,compensatory,number_generations,crossover_prob,mutation_prob,seed,dimensions))]
    if dimensions is False:    
        return average_fitness/(seed_end-seed_init+1)
    for i in range(len(average_dimensions)):
        average_dimensions[i]=average_dimensions[i]/(seed_end-seed_init+1)
    return average_dimensions


### Function that returns the fitness of the solution of a single student
def read_solution(student_row,domain_id,score_function,compensatory,number_generations,crossover_prob,mutation_prob,seed,dimensions):
    if compensatory:
        file_title="bestsol_"+str(student_row)+"_"+str(domain_id)+"_"+str(score_function)+"_comp_"+str(number_generations)+"_"+str(int(crossover_prob*100))+"_"+str(int(mutation_prob*100))+"_"+str(seed)
    else:
        file_title="bestsol_"+str(student_row)+"_"+str(domain_id)+"_"+str(score_function)+"_parcomp_"+str(number_generations)+"_"+str(int(crossover_prob*100))+"_"+str(int(mutation_prob*100))+"_"+str(seed)
    if os.path.exists(file_title):
        file=open(file_title,'r')       
        line  = file.readline()
        token_line=line.split(' ')
        if dimensions is False:
            file.close() 
            return float(token_line[len(token_line)-1])
        return float(token_line)
    return 0


### Function that returns the course effects
### Returns a 10x104 dataframe by default, where each row relates to each skill (10 soft skills)
### and 104 is the number of courses in the dataset
def get_courses_effects():
    #If the file exists, read it and return it
    if os.path.exists("./real_data/arranged_pooled_results_all.csv"):
        courses_effects=pd.read_csv("./real_data/arranged_pooled_results_all.csv")
        courses_effects.drop([courses_effects.columns[0],
                              courses_effects.columns[2],courses_effects.columns[3],
                              courses_effects.columns[4],courses_effects.columns[5],
                              courses_effects.columns[6],courses_effects.columns[111]],axis=1,inplace=True)
        return courses_effects
    return []


### Function that returns the student random intercept effects
### Returns a 10x1 dataframe by default, where each row relates to each skill (10 soft skills) of student with student_id
### 884  is the number of students in the dataset
def get_student_random_effect(student_id):
    #If the file exists, read it and return it
    if os.path.exists("./real_data/arranged_thetas_results_all.csv"):
        student_effects=pd.read_csv("./real_data/arranged_thetas_results_all.csv")
        return student_effects.iloc[:,student_effects.columns.get_loc("r_student["+str(student_id)+"]")]
    return []


### Function that returns the thresholds for the ordinal logistic regression
### Returns a 10x3 dataframe by default, where each row relates to each skill (10 soft skills) and 3 is the number of thresholds
def get_thresholds():
    #If the file exists, read it and return it
    if os.path.exists("./real_data/arranged_pooled_results_all.csv"):
        thresholds=pd.read_csv("./real_data/arranged_pooled_results_all.csv")
        return thresholds.iloc[:,[2,3,4]]
    return []


### Function that returns the thresholds for the minimum standard (for the moment mean sskill)
### Returns a 1x10 dataframe by default, where each column relates to each skill (10 soft skills) and 4 is the number of domains
def get_desired_standard(domain_id):
    #If the file exists, read it and return it
    if os.path.exists("./real_data/mean_skills_stage.csv"):
        descriptives_skills=pd.read_csv("./real_data/mean_skills_stage.csv")
        descriptives_skills.drop(descriptives_skills.columns[0],axis=1,inplace=True)
        return descriptives_skills.iloc[4+domain_id,[2,4,6,8,10,12,14,16,18,20]]
    return []


### Function that returns the courses by domain id
### Returns an unordered list of the courses elegible for domain id
def get_courses_domain(domain_id):
    #If the file exists, read it and return it
    if os.path.exists("./real_data/courses_ids_names_all.csv"):
        available_courses=pd.read_csv("./real_data/courses_ids_names_all.csv",encoding=('latin1'))
        available_courses.drop(available_courses.columns[0],axis=1,inplace=True)
        return [*available_courses.loc[available_courses["domain_id"]==domain_id]["variable_id"]]
    return []

#import math
#possible_courses=get_courses_domain(3)
#print(len(possible_courses))
#len_sol=11
#combs=math.comb(len(possible_courses),len_sol)
#print(combs)