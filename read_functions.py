# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:59:01 2023

@author: luis.pinos-ullauri
"""


import os.path
import pandas as pd


################## Reading Functions for the real data set ####################


### Function that returns the real data set
### Returns a pandas dataframe where each row belongs to a student in a particular stage
### The courses c1 to c11 describe the followed course identifiers up until that stage
def read_real_data():
    #If the file exists, read it and return it
    if os.path.exists("./real_data/full_dataset_coursesre.csv"):
        real_data=pd.read_csv("./real_data/full_dataset_coursesre.csv",encoding=('latin1'))
        real_data.drop([real_data.columns[0],real_data.columns[3],
                              real_data.columns[4],real_data.columns[5],
                              real_data.columns[6],real_data.columns[7],
                              real_data.columns[18],real_data.columns[19],
                              real_data.columns[20],real_data.columns[21],
                              real_data.columns[23],real_data.columns[24],
                              real_data.columns[25]],axis=1,inplace=True)
        return real_data
    return []


### Function that returns the course effects
### Returns a 10x104 pandas dataframe by default, where each row relates to each skill (10 soft skills)
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


### Function that returns the student random intercept effects (list)
### Returns a list by default, where each index corresponds to each skill (10 soft skills) of student with student_id
### 884  is the number of students in the dataset
def get_student_random_effect(student_id):
    #If the file exists, read it and return it
    if os.path.exists("./real_data/arranged_thetas_results_all.csv"):
        student_effects=pd.read_csv("./real_data/arranged_thetas_results_all.csv")
        return student_effects.iloc[:,student_effects.columns.get_loc("r_student["+str(student_id)+"]")].tolist()
    return []


### Function that returns the thresholds for the ordinal logistic regression model 
### The thresholds split the logit scale in 4 regions with K-1 values, where K is the number of possible responses
### Returns a 10x3 pandas dataframe by default, where each row relates to each skill (10 soft skills) and 3 is the number of thresholds
def get_thresholds():
    #If the file exists, read it and return it
    if os.path.exists("./real_data/arranged_pooled_results_all.csv"):
        thresholds=pd.read_csv("./real_data/arranged_pooled_results_all.csv")
        return thresholds.iloc[:,[2,3,4]]
    return []


### Function that returns the desired soft skill proficiency
### Returns a list by default, where each index corresponds to each skill (10 soft skills)
def get_desired_outcome(domain_id):
    #If the file exists, read it and return it
    if os.path.exists("./real_data/mean_skills_stage.csv"):
        descriptives_skills=pd.read_csv("./real_data/mean_skills_stage.csv")
        descriptives_skills.drop(descriptives_skills.columns[0],axis=1,inplace=True)
        return descriptives_skills.iloc[4+domain_id,[2,4,6,8,10,12,14,16,18,20]].tolist()
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


################## Reading Functions for the result files ####################


### Function that returns the appropiate directory address according to the parameters
### Returns a string to be used for the address of the result files
def toStringfilestructure(domain_id,compensatory,score_function=None,student_id=None,folder="./results/"):
    #Checking the tree structure for the file system
    if domain_id==1:
        domain_as_string="domain_ee/"
    elif domain_id==2:
        domain_as_string="domain_is/"
    elif domain_id==3:
        domain_as_string="domain_mx/"
    elif domain_id==4:
        domain_as_string="domain_nu/"
    if compensatory is True:
        compensatory_as_string="compensatory/"
    else:
        compensatory_as_string="partially_compensatory/"
    if score_function is None:
        return folder+domain_as_string+compensatory_as_string
    if score_function==1:
        score_function_as_string="linear/"
    elif score_function==2:
        score_function_as_string="logistic/"
    elif score_function==3:
        score_function_as_string="quadratic/"
    if student_id is not None:
        return folder+domain_as_string+compensatory_as_string+score_function_as_string+str(student_id)+"/"


### Function that reads the bestsol file and returns the fitness of the solution of a single student under a specific seed
### If dimensions is true. it will return a list of the overall fitness
def read_solution(student_id,domain_id,score_function,compensatory,number_generations,crossover_probability,mutation_probability,seed,dimensions):
    file_directory=toStringfilestructure(domain_id, compensatory, score_function, student_id)
    file_title=file_directory+"bestsol_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    #If the file exists, read it
    if os.path.exists(file_title):
        file=open(file_title,'r')
        #read the line
        line  = file.readline()
        #split the values
        token_line=line.split(' ')
        #return the overall fitness
        if dimensions is False:
            file.close() 
            return float(token_line[len(token_line)-11])
        #return the overall fitness with the soft skill scores
        #index of first soft skill score
        index_init=len(token_line)-11
        fitness_values=token_line[index_init:len(token_line)]
        fitness_values_float=[float(fitness_str) for fitness_str in fitness_values]
        if compensatory is False:
            overall_fitness=1
            for i in range(1,len(fitness_values_float)):
                overall_fitness=overall_fitness*fitness_values_float[i]
            fitness_values_float[0]=overall_fitness
        return fitness_values_float
    print(seed)
    print(student_id)


### Function that returns the average fitness from the different seeds (runs) of the same student
### If dimensions is true, it will return the average of the overall fitness alongside the average of the scores
### If dimensions is false, it will only return the average of the overall fitness
def read_average_solution(student_id,domain_id,score_function,compensatory,number_generations,crossover_probability,mutation_probability,seed_init,seed_end,dimensions):  
    #overall fitness
    average_fitness=0
    #list for soft skill scores
    average_dimensions=[0,0,0,0,0,0,0,0,0,0,0]
    #loop to check the seeds
    for seed in range(seed_init,seed_end+1):
        #calculate on overall fitness
        current_fitness=read_solution(student_id,domain_id,score_function,compensatory,number_generations,crossover_probability,mutation_probability,seed,dimensions)
        #print(current_fitness)
        if dimensions is False:
            average_fitness=average_fitness+current_fitness
        else:
            #print(average_dimensions)
            average_dimensions=[sum(x) for x in zip(average_dimensions,current_fitness)]
    if dimensions is False:    
        return average_fitness/(seed_end-seed_init+1)
    for i in range(len(average_dimensions)):
        average_dimensions[i]=average_dimensions[i]/(seed_end-seed_init+1)
    return average_dimensions


### Function that reads the bestsol file and returns the fitness of the solution of a single student under a specific seed
### If dimensions is true. it will return a list of the overall fitness
def read_fitness_by_gen(student_id,domain_id,score_function,compensatory,number_generations,crossover_probability,mutation_probability,seed):
    file_directory=toStringfilestructure(domain_id, compensatory, score_function, student_id)
    file_title=file_directory+"bestall_"+str(student_id)+"_"+str(number_generations)+"_"+str(int(crossover_probability*100))+"_"+str(int(mutation_probability*100))+"_"+str(seed)+".garcs"
    #If the file exists, read it
    if os.path.exists(file_title):
        file=open(file_title,'r')
        #read the line
        lines  = file.readlines()
        for i in range(len(lines)):
            #split the values
            token_line=lines[i].split(' ')
            if i==0:                
                fitness_gen_time=pd.DataFrame([0,float(token_line[0]),float(token_line[1])]).T
            else:
                fitness_gen_time=pd.concat([fitness_gen_time,pd.DataFrame([i,float(token_line[0]),float(token_line[1])]).T])
        return fitness_gen_time
    return 0


### Function that returns the average fitness from the different seeds (runs) of the same student
### If dimensions is true, it will return the average of the overall fitness alongside the average of the scores
### If dimensions is false, it will only return the average of the overall fitness
def read_average_fitness_by_gen(student_id,domain_id,score_function,compensatory,number_generations,crossover_probability,mutation_probability,seed_init,seed_end):
    #loop to check the seeds
    for seed in range(seed_init,seed_end+1):
        current_fitness_by_gen=read_fitness_by_gen(student_id, domain_id, score_function, compensatory, number_generations, crossover_probability, mutation_probability, seed)
        if seed==seed_init:
            average_fitness_by_gen=current_fitness_by_gen
        else:            
            for i in range(current_fitness_by_gen.shape[0]):
                average_fitness_by_gen.iloc[i,1]=average_fitness_by_gen.iloc[i,1]+current_fitness_by_gen.iloc[i,1]
                average_fitness_by_gen.iloc[i,2]=average_fitness_by_gen.iloc[i,2]+current_fitness_by_gen.iloc[i,2]
    for i in range(average_fitness_by_gen.shape[0]):
        average_fitness_by_gen.iloc[i,1]=average_fitness_by_gen.iloc[i,1]/(seed_end-seed_init+1)
        average_fitness_by_gen.iloc[i,2]=average_fitness_by_gen.iloc[i,2]/(seed_end-seed_init+1)
    return average_fitness_by_gen
