# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:09:35 2020

@author: STANDEL1
"""

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def crossover_nsga(num_variables, crossover_ratio, cross_fraction, pop, lb, ub):

    pop.iloc[:, num_variables:] = 0
    cross_over_pop = pop.copy()
    for i in range(0, len(pop), 2):
    
        # assign parents
        parent1 = cross_over_pop.iloc[i, :num_variables]
        parent2 = cross_over_pop.iloc[i+1, :num_variables]
   
        # assign Ratios
    
        crsFlag = np.random.uniform(size=(1, num_variables))[0] < cross_fraction
        randNum = np.random.uniform(low=0, high=1, size=(1, num_variables))[0]
    
        # create Children
    
        child1 = (np.array(parent1) + crsFlag * randNum * crossover_ratio * (np.array(parent2) - np.array(parent1)))
        child2 = (np.array(parent2) - crsFlag * randNum * crossover_ratio * (np.array(parent2) - np.array(parent1)))
    
        # Bounding Children
    
        child1 = children_bounding(child1, ub, lb)
        child2 = children_bounding(child2, ub, lb)
    
        # new crossed over population assignment
    

        cross_over_pop.iloc[i, :num_variables] = child1        
        cross_over_pop.iloc[i+1, :num_variables] = child2

    return cross_over_pop


def children_bounding(child, ub, lb):

    length = len(child)

    for i in range(length):
        child[i] = np.maximum(child[i], lb[i])
        child[i] = np.minimum(child[i], ub[i])
    
    return child
    
    
def mutation_nsga(num_variables, mutation_scale, mutation_shrink, mutation_fraction, pop, lb, ub, generation, num_generations):
    
    
    mutation_pop = pop.copy()
    for ind in range(len(pop)):
        
        parent = np.array(mutation_pop.iloc[ind,:num_variables])
        
        # calculating scale parameter
        
        scale = mutation_scale - mutation_shrink * mutation_scale * (generation + 1) / num_generations
        
        scale = scale * (ub - lb)
        
        # doing mutation
        
        child = parent.copy()
        
        for i in range(num_variables):
            if np.random.uniform() < mutation_fraction:
                child[i] = parent[i] + scale[i] * np.random.normal()
        
        
        # Bounding Children

        child = children_bounding(child, ub, lb)   
        
        # new mutated population assignment
        
        mutation_pop.iloc[ind,:num_variables] = child
    
    
    return mutation_pop

def tournament_selection(pop, useSurrogate, num_variables ):
    
    pop_size = len(pop)
    pool = np.zeros((1, pop_size), dtype=int)[0]
    
    randNum = np.random.randint(low = 0, high = pop_size, size = [1, 2*pop_size])[0]
    
    j = 0
    
    new_pop = pd.DataFrame(data = np.zeros((pop_size, len(pop.columns))), columns = pop.columns)
    
    if useSurrogate:
        rank = "pareto_front_surrogate"
        distance = "crowding_distance_surrogate"
    else:
        rank = "pareto_front"
        distance = "crowding_distance"
    
    for i in range(0, pop_size*2, 2):
        
        p1 = randNum[i]
        p2 = randNum[i+1]
        
        if (pop[rank][p1] < pop[rank][p2]) or (pop[rank][p1] == pop[rank][p2] and pop[distance][p1] > pop[distance][p2]):
            # p1 is better than p2
            new_pop.iloc[j,:] = pop.iloc[p1,:].values
            pool[j] = p1
            # print("P1 is better", pop[rank][p1], pop[rank][p2], p1)
            
        
        else:
            # p2 is better than p1 
            new_pop.iloc[j,:] = pop.iloc[p2,:].values
            pool[j] = p2
            # print("P2 is better", pop[rank][p2], pop[rank][p1], p2)
    
        j = j + 1
    
    return new_pop


def ndsort(pop, useSurrogate):
    
    pop_size = len(pop)
    
    Initial_np = np.zeros(pop_size, dtype = int)
    Initial_sp = [ [] for _ in range(pop_size) ]
    
    if useSurrogate:
        rank = "pareto_front_surrogate"
        distance = "crowding_distance_surrogate"
        Purity = "Purity_Surrogate"
        Recovery = "Recovery_Surrogate"
        nViol = "nViol_Surrogate"
        violSum = "violSum_Surrogate"
    else:
        rank = "pareto_front"
        distance = "crowding_distance"
        Purity = "Purity"
        Recovery = "Recovery"
        nViol = "nViol"
        violSum = "violSum"
        
    
    pop[rank] = 0
    pop[distance] = 0
    pop[distance] = pop[distance].astype("float32")
    
    # calculate domination Matrix
    
    nViol = np.array(pop[nViol], dtype = int)
    violSum = np.array(pop[violSum])
    obj = np.array(pop[[Purity, Recovery]])
    
    N = pop_size
    numObj = 2
    
    domMat = np.zeros(shape = (N,N), dtype = int)
    
    for p in range(N-1):
        for q in range(p+1, N):
            # 1. p and q are both feasible
            if (nViol[p] == 0 and nViol[q] == 0):
                pdomq = False;
                qdomp = False;
                for i in range(numObj):
                    if obj[p,i] < obj[q,i]:
                        pdomq = True
                    elif obj[p,i] > obj[q,i]:
                        qdomp = True
                
                if (pdomq and not qdomp):
                    domMat[p,q] = 1
                elif (not pdomq and qdomp):
                    domMat[p,q] = -1
            # 2. p is feasible and q is infeasible
            elif (nViol[p] == 0 and nViol[q] != 0):
                domMat[p,q] = 1
            # 3. q is feasible and p is infeasible
            elif (nViol[p] != 0 and nViol[q] == 0):
                domMat[p,q] = -1
            # 4. p and q are both feasible
            else:
                if violSum[p] < violSum[q]:
                    domMat[p,q] = 1
                elif violSum[p] > violSum[q]:
                    domMat[p,q] = -1
   
    domMat = np.array(np.matrix(domMat) - np.matrix(domMat).T)
    
    # compute np and sp for each individual
    
    for p in range(N-1):
        for q in range(p+1, N):
            if domMat[p,q] == 1: #p dominates q
                Initial_np[q] +=1
                Initial_sp[p] = np.append(Initial_sp[p], q)
            elif domMat[p,q] == -1:
                Initial_np[p] +=1
                Initial_sp[q] = np.append(Initial_sp[q], p)
    
    first_front = []
    
    # first front
    
    for i in range(N):
        if Initial_np[i] == 0:
            pop[rank][i] = 1
            first_front.append(i)
    

    fid = 1
    fid_indicator = True
    final_front = []
    final_front.append(first_front)
    intermediate_front = first_front
    
    while fid_indicator:
        Q = []
        for p in intermediate_front:
            
            for q in Initial_sp[int(p)]:
                
                Initial_np[int(q)] = Initial_np[int(q)] -1
                
                if Initial_np[int(q)] == 0:
                    pop[rank][int(q)] = fid + 1
                    Q.append(q)
        
        intermediate_front = Q
        fid += 1
        if len(Q) == 0:
            fid_indicator = False
        else:    
            final_front.append(intermediate_front)
        
    # Crowding Distance Calculation
    
    for fid in range(len(final_front)):
        idx = list(map(int,final_front[fid]))
        
        numInd = len(idx)
        
        obj = pop[[Purity,Recovery, distance]][pop[[Purity,Recovery, distance]].index.isin(idx)].reset_index()
        for m in range(numObj):
            if m == 0:
                objective = Purity
            else:
                objective = Recovery
            
            obj = obj.sort_values(by = [objective], ascending = [True]).reset_index(drop = True)

            # setting the first and last distance as infinity
            obj[distance][0] = np.inf
            obj[distance][numInd-1] = np.inf
            
            minobj = obj[objective][0]
            maxobj =  obj[objective][numInd-1]
            
            
            if numInd>2:
                for i in range(1, numInd-1):
                    if minobj == maxobj:
                        obj[distance][i] = obj[distance][i]
                        
                    else:
                        obj[distance][i] = obj[distance][i]  +  (obj[objective][i+1] - obj[objective][i-1])/(maxobj - minobj)
                    
            for el in idx:
                pop[distance][el] = obj[obj["index"] == el][distance].values
            
            
    return pop    
        
        
                    
    
    
    
    
    
    



