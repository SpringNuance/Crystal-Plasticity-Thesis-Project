import numpy as np

def D1(exp_stress, sim_stress): 
    return np.sqrt(np.sum(np.square(exp_stress - sim_stress))/np.sum(np.square(exp_stress)))

def D2(exp_stress, sim_stress, strainobj): # Added strainobj as an input
    exp_stress_d1 = np.diff(exp_stress)/np.diff(strainobj)  
    sim_stress_d1 = np.diff(sim_stress)/np.diff(strainobj)
    return np.sqrt(np.sum(np.square(sim_stress_d1 - exp_stress_d1))/np.sum(np.square(exp_stress_d1)))

def D3(exp_stress, sim_stress):
    return np.max(np.sqrt(np.square(exp_stress - sim_stress)/sum(np.square(exp_stress))))

def D4(exp_stress, sim_stress, strainobj):
    exp_stress_d1 = np.diff(exp_stress)/np.diff(strainobj)
    sim_stress_d1 = np.diff(sim_stress)/np.diff(strainobj)
    return np.max(np.sqrt(np.square(sim_stress_d1 - exp_stress_d1)/np.sum(np.square(exp_stress_d1))))

def closeYield(exp_stress, sim_stress):
    expYieldStress = exp_stress[0]
    simYieldStress = sim_stress[0]
    upper = expYieldStress * 1.02
    lower = expYieldStress * 0.98 
     
    if simYieldStress >= lower and simYieldStress <= upper:
        return 0
    else:
        return 1

def chromosomefitness(exp_stress, sim_stress,strainobj, w1, w2, w3, w4):
    return ( w1*D1(exp_stress, sim_stress) + w2*D2(exp_stress, sim_stress, strainobj) 
            + w3*D3(exp_stress, sim_stress) + w4*D4(exp_stress, sim_stress, strainobj) + closeYield(exp_stress, sim_stress))

w1 = 0.9
w2 = 0.005
w4 = 0.009
w3 = 1 - w1 - w2 - w4

def insideFivePercentStd(exp_target, sim_stress):
    upperStress = 1.05 * exp_target
    lowerStress = 0.95 * exp_target
    for i in range(exp_target.size):
        if sim_stress[i] < lowerStress[i] or sim_stress[i] > upperStress[i]:
            return False 
    return True