# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:34:29 2020

@author: STANDEL1
"""

import matlab.engine
import pandas as pd
import numpy as np
import Helper_function_matlab as hf
from time import time
import numpy.random as npr
from sklearn.ensemble import RandomForestRegressor

#from multiprocessing import Pool

#from simulation_functions_Parallel_Systems import sim_process_stable

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

#npr.seed(1)

start = time()

eng = matlab.engine.start_matlab()
eng.ProcessOptimization_Python(nargout=0)

IsothermParams = eng.workspace['IsothermParams']
material = eng.workspace['material']
type_ = eng.workspace['type']
N = eng.workspace['N']


#lb = matlab.double([1e5,  10, 0.01, 0.1, 0, 1e4])
#ub = matlab.double([10e5, 1000, 0.99, 2, 1, 5e4])

#x = lb

#a,b = eng.PSACycleSimulation(x, material, type_,N,nargout=2)


#Initialization of Variables
num_variables = 6
sol_per_pop = int(60/1)
pop_init_size = int(600/1)
pop_test_size = int(200/1)

#Setting the Number of generations and the ML Limit & mutation rate
num_generations = 60 #50
mutation_rate = 0.3
elite_rate = 0.15

crossover_ratio = 1.2
cross_fraction = 2/num_variables 

mutation_scale = 0.1
mutation_shrink = 0.5
mutation_fraction = 2/num_variables

useSurrogate = True
timing = True
elite_size = int(sol_per_pop*elite_rate)

gen_statistics_list = []
population_details = pd.DataFrame()

Surr_Acc = []
x = 0

def sim_process_population(population):
    pop_size = len(population)
    
    Revenue_Array = pd.DataFrame(np.c_[
        np.zeros(pop_size),
        np.zeros(pop_size),
        np.zeros(pop_size),
        np.zeros(pop_size),
        np.zeros(pop_size),
        np.zeros(pop_size),
        np.zeros(pop_size),
        ], columns = ["Purity", "Recovery", "Constraint_1", "Constraint_2", "Constraint_3", "nViol", "violSum"])
        
    for i, (x_0,x_1,x_2,x_3,x_4,x_5) in enumerate(population.values[:,0:num_variables]):
        print("Evaluating:", i)
        a,b = eng.PSACycleSimulation(matlab.double([x_0,  x_1, x_2, x_3, x_4, x_5]), material, type_,N, nargout=2)
        print(a)
        Revenue_Array.Purity[i] = a[0][0]
        Revenue_Array.Recovery[i] = a[0][1]
        Revenue_Array.Constraint_1[i] = b[0][0]
        Revenue_Array.Constraint_2[i]  = b[0][1]
        Revenue_Array.Constraint_3[i]  = b[0][2]
        count = 0
        if b[0][0]>0:
            count+=1
        if b[0][1]>0:
            count+=1
        if b[0][2]>0:
            count+=1
        Revenue_Array.nViol[i]  = count
        Revenue_Array.violSum[i] = abs(b[0][0]) + abs(b[0][1]) + abs(b[0][2])
    
    return Revenue_Array
        
    


def gather_statistics(gen_statistics, population, useSurrogate):
    #Gather statistics
    if useSurrogate:
        ranking = population.sort_values(by=["pareto_front_surrogate", "crowding_distance_surrogate"], ascending=[True, False])
        # elite = ranking.iloc[:elite_size,:]
        gen_statistics = gen_statistics.append(ranking)
        # gen_statistics.loc[generation,"Mean_Revenue_Surrogate"] = population.Revenue_Surrogate.mean()
        # gen_statistics.loc[generation,"Mean_Revenue_Elite_Surrogate"] = elite.Revenue_Surrogate.mean()
        # gen_statistics.loc[generation,"Max_Revenue_Surrogate"] = population.Revenue_Surrogate.max()

    ranking = population.sort_values(by=["pareto_front", "crowding_distance"], ascending=[True, False])
    # elite = ranking.iloc[:elite_size,:]
    gen_statistics = gen_statistics.append(ranking)
    # gen_statistics.loc[generation,"Mean_Revenue"] = population.Revenue.mean()
    # print(population.Revenue.mean())
    # gen_statistics.loc[generation,"Mean_Revenue_Elite"] = elite.Revenue.mean()
    # gen_statistics.loc[generation,"Max_Revenue"] = population.Revenue.max()


for _ in range(10):
    

    
    #Lower & Upper Bounds of Decision Variables
    lb = np.array([1e5,  10, 0.01, 0.1, 0, 1e4])
    ub = np.array([10e5, 1000, 0.99, 2, 1, 5e4])
    #Creating the initial population
    new_population_x_0 = npr.randint(1e5, 10e5, pop_init_size + pop_test_size)
    new_population_x_1 = npr.uniform(10, 1000, pop_init_size + pop_test_size)
    new_population_x_2 = npr.uniform(0.01,  0.99,  pop_init_size + pop_test_size)
    new_population_x_3 = npr.uniform(0.1, 2, pop_init_size + pop_test_size)
    new_population_x_4 = npr.uniform(0, 1, pop_init_size + pop_test_size)
    new_population_x_5 = npr.uniform(1e4, 5e4, pop_init_size + pop_test_size)

    #Joining the initial population
    new_pop = pd.DataFrame(np.c_[
        new_population_x_0,
        new_population_x_1,
        new_population_x_2,
        new_population_x_3,
        new_population_x_4, 
        new_population_x_5,
        np.zeros(pop_init_size + pop_test_size),
        np.zeros(pop_init_size + pop_test_size),
        np.zeros(pop_init_size + pop_test_size),
        np.zeros(pop_init_size + pop_test_size),
        np.zeros(pop_init_size + pop_test_size),
        np.zeros(pop_init_size + pop_test_size)
        ], columns = ["pop1_x0", "pop1_x1","pop1_x2","pop1_x3","pop1_x4","pop1_x5","Purity", "Recovery","nViol", "violSum", "pareto_front", "crowding_distance"])

    
    #evaluating population
    df = sim_process_population(new_pop)
    new_pop.Purity = df.Purity.values
    new_pop.Recovery = df.Recovery.values
    new_pop.nViol = df.nViol.values
    new_pop.violSum = df.violSum.values
    
    
    #calculating pareto front and crowding distance    
    new_pop = hf.ndsort(new_pop, useSurrogate = False)

    
    #assigning the test set for the surrogate model
    test_pop = new_pop.iloc[:pop_test_size,:]
    new_pop  = new_pop.iloc[pop_test_size:,:]

    
    ranking = new_pop.sort_values(by=["pareto_front", "crowding_distance"], ascending=[True, False])


    #Starting the Total_pops at the above new_pop created as initial
    population = ranking[0:sol_per_pop].copy().reset_index(drop = True)
    total_pop = new_pop.copy().reset_index(drop = True)

    #target setting for a ML
    X = total_pop.iloc[:,0:-6]
    y_Purity = total_pop.iloc[:,-6]
    y_Recovery = total_pop.iloc[:,-5]

    
    X_test = test_pop.iloc[:,0:-6]
    
    y_Purity_test = test_pop.iloc[:,-6]
    y_Recovery_test = test_pop.iloc[:,-5]
    
    #Using Random Forest as Model for both fitness functions
    rf_Purity = RandomForestRegressor()
    rf_Recovery = RandomForestRegressor()
    #Model training
    rf_Purity.fit(X, y_Purity)
    rf_Recovery.fit(X, y_Recovery)
    print("Purity Surrogate Accuracy", rf_Purity.score(X_test,y_Purity_test))
    print("Recovery Surrogate Accuracy", rf_Recovery.score(X_test,y_Recovery_test))


    if useSurrogate:
        population["Purity_Surrogate"] = rf_Purity.predict(population.iloc[:,0:num_variables])
        population["Recovery_Surrogate"] = rf_Recovery.predict(population.iloc[:,0:num_variables])
        
        population.loc[abs(population["Recovery_Surrogate"]) < 0.9, 'nViol_Surrogate'] = 1
        population.loc[abs(population["Recovery_Surrogate"]) >= 0.9, 'nViol_Surrogate'] = 0
        
        population.loc[abs(population["Recovery_Surrogate"]) < 0.9, 'violSum_Surrogate'] = 0.9 - abs(population["Recovery_Surrogate"])
        population.loc[abs(population["Recovery_Surrogate"]) >= 0.9, 'violSum_Surrogate'] = 0

        population = hf.ndsort(population, useSurrogate)
        
        total_pop["Purity_Surrogate"] = rf_Purity.predict(total_pop.iloc[:,0:num_variables])
        total_pop["Recovery_Surrogate"] = rf_Recovery.predict(total_pop.iloc[:,0:num_variables])
        
        total_pop.loc[abs(total_pop["Recovery_Surrogate"]) < 0.9, 'nViol_Surrogate'] = 1
        total_pop.loc[abs(total_pop["Recovery_Surrogate"]) >= 0.9, 'nViol_Surrogate'] = 0
        
        total_pop.loc[abs(total_pop["Recovery_Surrogate"]) < 0.9, 'violSum_Surrogate'] = 0.9 - abs(total_pop["Recovery_Surrogate"])
        total_pop.loc[abs(total_pop["Recovery_Surrogate"]) >= 0.9, 'violSum_Surrogate'] = 0

        total_pop = hf.ndsort(total_pop, useSurrogate)
        
        
    gen_statistics = pd.DataFrame()

    gen_statistics_list.append(gen_statistics)

    for generation in range(num_generations):
        print("Generation", generation)

        if useSurrogate:
            ranking = population.sort_values(by=["pareto_front_surrogate", "crowding_distance_surrogate"], ascending=[True, False])
        else:
            ranking = population.sort_values(by=["pareto_front", "crowding_distance"], ascending=[True, False])

        #Select the next population to be sent to tournament selection
        pop = ranking.iloc[:sol_per_pop,:].reset_index(drop = True)
        
        
        #Generate Statitics
        if not timing:
            gather_statistics(gen_statistics, population, useSurrogate)
        population_details = population_details.append(pop)
        
        #create the population for crossover and mutation           
        new_population = hf.tournament_selection(pop, useSurrogate, num_variables)

        #Executing Crossover
        new_population = hf.crossover_nsga(num_variables, crossover_ratio, cross_fraction, new_population, lb, ub)
        #Executing Mutation
        new_population = hf.mutation_nsga(num_variables, mutation_scale, mutation_shrink, mutation_fraction, new_population, lb, ub, generation, num_generations)
        
        
        #Evaluate the new_population
        retrain = False
        if useSurrogate:
            new_population["Purity_Surrogate"] = rf_Purity.predict(new_population.iloc[:,0:num_variables])
            new_population["Recovery_Surrogate"] = rf_Recovery.predict(new_population.iloc[:,0:num_variables])
            
            new_population.loc[abs(new_population["Recovery_Surrogate"]) < 0.9, 'nViol_Surrogate'] = 1
            new_population.loc[abs(new_population["Recovery_Surrogate"]) >= 0.9, 'nViol_Surrogate'] = 0
        
            new_population.loc[abs(new_population["Recovery_Surrogate"]) < 0.9, 'violSum_Surrogate'] = 0.9 - abs(new_population["Recovery_Surrogate"])
            new_population.loc[abs(new_population["Recovery_Surrogate"]) >= 0.9, 'violSum_Surrogate'] = 0
            
            
            new_population = hf.ndsort(new_population, useSurrogate)
            
            
            
            ranking_surr = new_population.sort_values(by=["pareto_front_surrogate", "crowding_distance_surrogate"], ascending=[True, False])
            elite_surr =  ranking_surr.iloc[:elite_size,:]
            
            df = sim_process_population(elite_surr)
            elite_surr_actual_Recovery = df.Recovery.values             
            elite_surr_actual_Purity = df.Purity.values 
            

            if np.abs(elite_surr_actual_Purity.mean() - elite_surr.Purity_Surrogate.mean()) > elite_surr_actual_Purity.std() or np.abs(elite_surr_actual_Recovery.mean() - elite_surr.Recovery_Surrogate.mean()) > elite_surr_actual_Recovery.std():
                retrain = True

        if retrain or not timing or not useSurrogate:
            df = sim_process_population(new_population)
            new_population.Purity = df.Purity.values
            new_population.Recovery = df.Recovery.values
            new_population.nViol = df.nViol.values
            new_population.violSum = df.violSum.values

        if retrain:
            total_pop = total_pop.append(new_population, ignore_index=True)
            X = total_pop.iloc[:,0:num_variables]
            y_Purity = total_pop.Purity
            y_Recovery = total_pop.Recovery
            rf_Purity.fit(X,y_Purity)
            rf_Recovery.fit(X,y_Recovery)
            print("Purity Surrogate Accuracy", rf_Purity.score(X_test,y_Purity_test))
            print("Recovery Surrogate Accuracy", rf_Recovery.score(X_test,y_Recovery_test))
            #population["Revenue_Surrogate"] = rf.predict(population.iloc[:,0:num_variables])
            #Surr_Acc.append(rf.score(X_test,y_test))
        
        
        
        
        #Combine Population created from crossover and mutation with population before
        population = pop
        population = population.append(new_population, ignore_index = True)
        
        
        #calculate the pareto front ranking and the crowding distance of the combined population
        population = hf.ndsort(population, useSurrogate)

        if generation == 10 or generation == 20 or generation == 30 or generation == 40 or generation == 59:
            population_details.to_csv("Final_Results_Surrogate_Timed_7" + str(generation) + ".csv")



stop = time()
print("Elapsed time:", stop, start)
print("Elapsed time during the whole program in seconds:", stop-start)
#pd.concat(gen_statistics_list).to_csv("Parallel_System_simulation_30_exp.csv")
population_details.to_csv("Final_Results_Surrogate_Timed_7.csv")


