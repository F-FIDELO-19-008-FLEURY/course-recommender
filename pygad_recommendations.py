# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:56:04 2023

@author: luis.pinos-ullauri
"""

import read_functions as rdf
import numpy as np
import random as rand
import math
from scipy.special import expit
import pygad

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
            fitness=fitness+score
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

### Function that performs a single point crossover between two parent combinations
### maintaining the constraint of different followed courses
def spcrossover_func(parents, offspring_size, ga_instance):  
    #print("crossover")
    #print("offspring size")
    #print(offspring_size)    
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        #if parent1.size != np.unique(parent1).size or parent2.size != np.unique(parent2).size:
        #    print("PROBLEEEEEEEEEEEEEEEEEEEEEEEM IN CROSSSSS BEG")
        #    print(parent1)
        #    print(parent2)
        #if len(parent1) != len(set(parent1)):
            #print(parent1)
            #print(idx)
            #print(offspring)
        #if len(parent2) != len(set(parent2)):
        #    print(parent2)
        parent1.sort()
        parent2.sort()
        #print("parents sorted")
        #print(parent1)
        #print(parent2)
        split_point=solve_duplicates(parent1, parent2)
        #print("split point")
        #print(split_point)
        child=parent1.copy()
        child[split_point+1:]=parent2[split_point+1:]
        #parent1[split_point+1:] = parent2[split_point+1:]
        #print("child")
        #print(parent1)
        #if child.size != np.unique(child).size:
        #    print("PROBLEEEEEEEEEEEEEEEEEEEEEEEM IN CROSSSSS AFTER")
        #    print("parents sorted")
        #    print(parent1)
        #    print(parent2)
        #    print("split point")
        #    print(split_point)
        #    print("child")
        #    print(child)            
        offspring.append(child)

        idx += 1
    
    #print(len(offspring))

    return np.array(offspring)

### Function that performs uniform crossover between two parent combinations
### maintaining the constraint of different followed courses
def unifcrossover_func(parents, offspring_size, ga_instance):  
    #print("crossover")
    weights=[ga_instance.crossover_probability,1-ga_instance.crossover_probability]
    outcomes=[1,0]
    #print("offspring size")
    #print(offspring_size)    
    offspring = []
    idx = 0
    #print(parents.shape[0])
    #print("crossover")
    while len(offspring) != offspring_size[0]:
        #print("check")
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        #if parent1.size != np.unique(parent1).size or parent2.size != np.unique(parent2).size:
        #    print("PROBLEEEEEEEEEEEEEEEEEEEEEEEM IN CROSSSSS BEG")
        #    print(parent1)
        #    print(parent2)
        #if len(parent1) != len(set(parent1)):
            #print(parent1)
            #print(idx)
            #print(offspring)
        #if len(parent2) != len(set(parent2)):
        #    print(parent2)
        #parent1.sort()
        #parent2.sort()
        #print("parents sorted")
        #print(parent1)
        #print(parent2)
        #split_point=solve_duplicates(parent1, parent2)
        #print("split point")
        #print(split_point)
        child=parent2.copy()#child is copy of parent2 by default
        if rand.choices(outcomes,weights)==[1]:
            for i in range(len(parent1)):
                if not np.where(parent1[i]==parent2)[0].size!=0:#gene parent1[i] not in parent[2]
                    if rand.random()<0.5:#copy parent1
                        #print("copy parent1")
                        child[i]=parent1[i]
        #print(child)
                    
        
        
        #child[split_point+1:]=parent2[split_point+1:]
        #parent1[split_point+1:] = parent2[split_point+1:]
        #print("child")
        #print(parent1)
        #if child.size != np.unique(child).size:
        #    print("PROBLEEEEEEEEEEEEEEEEEEEEEEEM IN CROSSSSS AFTER")
        #    print("parents sorted")
        #    print(parent1)
        #    print(parent2)
        #    print("split point")
        #    print(split_point)
        #    print("child")
        #    print(child)            
        offspring.append(child)

        idx += 1
    
    #print(len(offspring))

    return np.array(offspring)

### Functions that performs a single point mutation on one course
### It checks whether the new course is already in the combination and if so
### Generates a new one
def mutation_func(offspring, ga_instance):
    #fitness_values=ga_instance.previous_generation_fitness
    #mean_fitness=sum(fitness_values)/len(fitness_values)
    #print(mean_fitness)
    weights=[ga_instance.mutation_probability,1-ga_instance.mutation_probability]
    outcomes=[1,0]
    #print(weights)
    #print(outcomes)
    #print(rand.choices(outcomes,weights)==[1])
    for chromosome_idx in range(offspring.shape[0]):
        if rand.choices(outcomes,weights)==[1]:
            #print("mutation")
            flag=False
            count=0
            random_gene=np.random.choice(possible_courses)
            random_gene_idx = np.random.choice(range(offspring.shape[1]))
            solution=offspring[chromosome_idx].copy()
            #print(solution)
            #print(random_gene_idx)        
            while flag!=True and count<=10:        
                index=np.where(solution==random_gene)
                if index[0].size!=0:#random gene is already in the current chromosome
                    #print("random gene is in current chromosome")  
                    #print(offspring[chromosome_idx])
                    #print(random_gene)
                    random_gene=np.random.choice(possible_courses)
                    count+=1
                    #print("try again")
                else:#new random gene is not in the current chromosome
                    #print("new random gene is not in current chromosome")
                    #print(offspring[chromosome_idx])
                    #print(random_gene)
                    #print(random_gene_idx)
                    solution[random_gene_idx]=random_gene
                    #if solution.size != np.unique(solution).size:
                    #    print("PROBLEEEEEEEEEEEEEEEEEM IN MUTTTTTTTTTT")
                    #    print(random_gene)
                    #    print(random_gene_idx)
                    #    print(offspring[chromosome_idx])
                    #    print(solution)
                    #print("mutated chromosome")
                    offspring[chromosome_idx]=solution
                    flag=True
            #print("after mutation")
            #print(solution)
            #offspring[chromosome_idx].sort()
            #if count==5:
            #    print("MAX")

    return offspring

### Function that returns the appropiate index for the cutpoint between two solutions a1 and a2
### It is used in the crossover function to avoid duplicates in the combinations
def solve_duplicates(a1,a2):
    i=0
    index=-1
    #for each element within array a1
    for i in range(len(a1)):
        #if a1[i] is within a2
        if np.where(a1[i]==a2)[0].size!=0 and np.where(a1[i]==a2)[0][0]>=index:
            index=np.where(a1[i]==a2)[0][0]
    if index==len(a1)-1:
        return -1
    if index!=-1:
        return index
    return np.random.choice(range(len(a1)))

### Function that saves a combination of courses into the appropiate cells at the dataframe
def save_recommendations(recommendations,row,combination,N_courses_followed):
    for i in range(N_courses_followed):
        recommendations.iloc[row,i]=combination[i]


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



################## End of  User Defined Functions #############################

### Through multiple trials, convergence of the GA is estimated to be around 100 generations
### EXPLANATION: The random effects of the courses, whose mean is approximately 0
### It would be a differenet with fixed effects for each course

"""
#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
#real_data_stage2=real_data_stage2.loc[real_data_stage2["domain_id"]==1]
real_data_stage2=real_data_stage2.reset_index(drop=True)
recommendations=pd.DataFrame(np.zeros(shape=(len(real_data_stage2),14)))
N_executions=1
#Thresholds
thresholds=rdf.get_thresholds()
#Course Effects
courses_effects=rdf.get_courses_effects()
#Minimum Soft skill proficiency
min_skill=1
#Maximum Soft skill proficiency
max_skill=4
#Compensatory boolean variable
comp=True
#Score function flag variable
#score_func=1#Linear
#score_func=2#Quadratic
score_func=3#Exponential


for i in range(len(real_data_stage2)):
    print(i)
    student_id=real_data_stage2.iloc[i,0]
    domain_id=real_data_stage2.iloc[i,26]
    N_courses_followed=real_data_stage2.iloc[i,25]
    #Student Effect
    theta=rdf.get_student_random_effect(student_id)
    #Desired outcome
    desired_outcome=rdf.get_desired_standard(domain_id)
    #get possible courses
    possible_courses=rdf.get_courses_domain(domain_id)    
    fitness_function = fitness_func
    num_generations = 100
    num_parents_mating = 100
    sol_per_pop = 100
    num_genes = int(N_courses_followed)
    parent_selection_type = "tournament"
    mutation_percent_genes = 10
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           crossover_type=crossover_func,
                           mutation_type=mutation_func,
                           crossover_probability=0.75,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           allow_duplicate_genes=False,
                           parent_selection_type=parent_selection_type,                       
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=int,gene_space=possible_courses,
                           keep_elitism=1,suppress_warnings=True)
    #print(ga_instance.initial_population)
    for j in range(N_executions):
        ga_instance.run()
        #ga_instance.plot_fitness()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        if j==0:
            recommendations.iloc[i,11]=solution_fitness
            #print("Parameters of the best solution : {solution}".format(solution=solution))
            #print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        elif recommendations.iloc[i,11]<solution_fitness:
            recommendations.iloc[i,11]=solution_fitness
            #print("New Parameters of the best solution : {solution}".format(solution=solution))
            #print("New Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    recommendations.iloc[i,11]=solution_fitness
    recommendations.iloc[i,12]=student_id
    recommendations.iloc[i,13]=domain_id
    save_recommendations(recommendations,i,solution,N_courses_followed)

recommendations.columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10',
                         'c11','fitness','student_id','domain_id']

#recommendations.to_csv("./real_data/recommendations.csv")

if score_func==1 and comp:
    recommendations.to_csv("./real_data/recommendations_lin_comp.csv")
if score_func==1 and not comp:
    recommendations.to_csv("./real_data/recommendations_lin_parcomp.csv")
if score_func==2 and comp:
    recommendations.to_csv("./real_data/recommendations_quad_comp.csv")
if score_func==2 and not comp:
    recommendations.to_csv("./real_data/recommendations_quad_parcomp.csv")
if score_func==3 and comp:
    recommendations.to_csv("./real_data/recommendations_exp_comp.csv")
if score_func==3 and not comp:
    recommendations.to_csv("./real_data/recommendations_exp_parcomp.csv")

"""
#Read real data set
real_data=rdf.read_real_data()
#Considering only stage 2
real_data_stage2=real_data.loc[real_data["stage"]==2]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]>5]
real_data_stage2=real_data_stage2.loc[real_data_stage2["N_courses_followed"]<12]
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

#low_mutation_probability=0.1
#high_mutation_probability=0.3
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
fitness_function = fitness_func
num_generations = 500
num_parents_mating = 50
sol_per_pop = 100
num_genes = int(N_courses_followed)
parent_selection_type = "tournament"
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       crossover_type=unifcrossover_func,
                       mutation_type=mutation_func,
                       crossover_probability=0.65,
                       mutation_probability=0.15,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       allow_duplicate_genes=False,
                       parent_selection_type=parent_selection_type,
                       gene_type=int,gene_space=possible_courses,
                       keep_parents=1,
                       keep_elitism=1,
                       suppress_warnings=True)
check=ga_instance.initial_population
check.sort()
print("unique courses in the initial population",len(np.unique(check)))
#print(check)
ga_instance.run()
ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution.sort()
print("Parameters of the best solution :",solution)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Crossover probability:",ga_instance.crossover_probability)
print("Mutation probability:",ga_instance.mutation_probability)
print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")











########### TO DO

### WHICH COURSES CAN BE CHOSEN BY WHICH SPECIALISATION, IN THAT WAY, THE GENE SPACE CAN BE DETERMINED CORRECTLY
### COURSE 103 IS NOT CONSIDERED IN THE DATASET, IT JUMPS FROM 102 TO 104
### AND MAYBE PERFORM THE RECOMMENDATIONS SEPARATELY. BUT WOULD THEN NEED TO CONSTRAINT WHICH COURSES CAN BE CHOSEN BY WHICH SPECIALISATION
### CHECK HOW THE RECOMMENDATIONS FARE BETWEEN THE DIFFERENT SPECIALISATIONS AND THE DIFFERENT SCORING AND COMPENSATION FUNCTIONS
### SHOW FITNESS PLOTS ACROSS GENERATIONS
### EMPIRICALLY SELECT CROSSOVER AND MUTATION RATE
### COMPARE RESULTS IN DIFFERENT SPECIALISATIONS
### NEED TO RUN MULTIPLE TIMES THE ALGORITHM FOR EACH STUDENT AND SELECT THE COURSE SET WITH THE BEST FITNESS
### EMPIRICALLY ESTIMATE BRUTE FORCE TIME
### COMPARISONS OF ALL COMBINATIONS








