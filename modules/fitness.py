import numpy as np
from math import *

def D1(exp_stress, sim_stress): 
    return np.sqrt(np.sum(np.square(exp_stress - sim_stress))/np.sum(np.square(exp_stress)))

def D2(exp_stress, sim_stress, sim_strain): 
    exp_stress_d1 = np.diff(exp_stress)/np.diff(sim_strain)  
    sim_stress_d1 = np.diff(sim_stress)/np.diff(sim_strain)
    return np.sqrt(np.sum(np.square(sim_stress_d1 - exp_stress_d1))/np.sum(np.square(exp_stress_d1)))

def D3(exp_stress, sim_stress):
    return np.max(np.sqrt(np.square(exp_stress - sim_stress)/sum(np.square(exp_stress))))

def D4(exp_stress, sim_stress, sim_strain):
    exp_stress_d1 = np.diff(exp_stress)/np.diff(sim_strain)
    sim_stress_d1 = np.diff(sim_stress)/np.diff(sim_strain)
    return np.max(np.sqrt(np.square(sim_stress_d1 - exp_stress_d1)/np.sum(np.square(exp_stress_d1))))

def fitness_yield(exp_stress, sim_stress):
    expYieldStress = exp_stress[0]
    simYieldStress = sim_stress[0] 
    return abs(expYieldStress - simYieldStress)

def fitness_hardening(exp_stress, sim_stress, sim_strain, w1, w2, w3, w4):
    return ( w1*D1(exp_stress, sim_stress) + w2*D2(exp_stress, sim_stress, sim_strain) 
            + w3*D3(exp_stress, sim_stress) + w4*D4(exp_stress, sim_stress, sim_strain))

def insideYieldStressDev(exp_stress, sim_stress, percentDeviation):
    expYieldStress = exp_stress[0]
    simYieldStress = sim_stress[0] 
    upper = expYieldStress * (1 + percentDeviation * 0.01) 
    lower = expYieldStress * (1 - percentDeviation * 0.01) 
    if simYieldStress >= lower and simYieldStress <= upper:
        return True
    else:
        return False

def insideHardeningDev(exp_target, sim_stress, percentDeviation):
    upperStress = exp_target * (1 + percentDeviation * 0.01) 
    lowerStress = exp_target * (1 + percentDeviation * 0.01) 
    for i in range(exp_target.size):
        if sim_stress[i] < lowerStress[i] or sim_stress[i] > upperStress[i]:
            return False 
    return True