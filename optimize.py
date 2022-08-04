# External libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
# Our classes
from modules.SIM import *
from modules.preprocessing import *
from modules.fitness import *
from modules.param_ranges import *
from modules.helper import *
from optimization.GA import *
from optimization.BA import *
from os import path

###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

# -------------------------------------------------------------------
#   Stage 0: Choose the CP model, the optimization algorithm, number of initial simulations,
#   the experimental curve index to fit (1,2,3), project path folder and material name
# -------------------------------------------------------------------

# Type "PH" for phenomenological law
# Type "DB" for dislocation-based law
CPLaw = "PH" # Please change this

# Type "GA" for genetic algorithm
# Type "BA" for Bayesian algorithm
algorithm = "GA" # Please change this

# Please choose the number of initial simulations
initialSims = 30 # Please change this

# Type the target experimental curve index (1,2,3)
curveIndex = 1 # Please change this

# Type the project path folder
# projectPath = "/scratch/project_2004956/Binh/CrystalPlasticityProject"
projectPath = "/scratch/project_2004956/Binh/DB1GeneticLargeRVE"

# Type the material name
material = "RVE_1_40_D"
# material = "512grains512"

# Type automatic or manual methods
method = "auto"
# method = "manual"

if material == "RVE_1_40_D":
    param_ranges = param_ranges_RVE_1_40_D
    default_yield_values = default_yield_RVE_1_40_D 
    # These number are obtained from material.config files, where each number is the line number of the parameter minus 1
    # For example, the line of the parameter alpha in RVE_1_40_D in the material.config file template is 34 => 33
    # Lines of   alpha, h0, tau0, taucs 
    editlinesPH = [33,  34,  31,   32]
    # Lines of   dipole, islip, omega, p,  q, tausol
    editLinesDB = [66,    62,    65,  58,  59,  49]
    # This RVE is large so needs to run heavy job files
    mode = "heavy"
elif material == "512grains512":
    param_ranges = param_ranges_512grains512
    default_yield_values = default_yield_512grains512 
    # Lines of   alpha, h0, tau0, taucs 
    editlinesPH = [54,  46,  36,   37]
    # Lines of   dipole, islip, omega, p,  q, tausol
    editLinesDB = [48,    44,    47,  40,  41,  31]
    # This RVE is small so needs to run light job files
    mode = "light"

param_range = param_ranges[CPLaw][curveIndex - 1] # param_range is used for SIM object
param_range_no_round = param_range_no_round_func(param_range) # param_range_no_round is used to be fed to GA
param_range_no_step = param_range_no_step_func(param_range_no_round) # param_range_no_step is used to be fed to BA
default_yield_value = default_yield_values[CPLaw][curveIndex - 1]
# print("param_range is:")
# print(param_range)
# print("param_no_round is:")
# print(param_range_no_round)
# print("param_range_no_step is:")
# print(param_range_no_step)

if CPLaw == "PH":
    numberOfParams = 4
    convertUnit = 1
    # tau0 is the main param affecting yield stress in PH model
elif CPLaw == "DB":
    numberOfParams = 6
    convertUnit = 1e-6 # In DAMASK simulation, stress unit is Pa instead of MPa in DB model
    # p, q, tausol is the main param affecting yield stress in DB model

print("Welcome to Crystal Plasticity Parameter Calibration")
print("The configurations you have chosen: ")
print("CP Law:", CPLaw)
target_curve = f"{CPLaw}{curveIndex}"
print("The target curve:", target_curve)
print("Number of fitting parameters in", CPLaw, "law:", numberOfParams)
print("Range and step of parameters: ")
print(param_range_no_round)
print("Default values of hardening parameters for yield stress optimization:")
print(default_yield_value)
print("Number of initial simulations:", initialSims)
print("Chosen optimization algorithm:", algorithm)
print("Material under study:",material)
print("The optimization process is", method)
print("The path to your main project folder is: ")
print(projectPath)
# Initialize the SIM object.
info = {
    'param_range': param_range,
    'CPLaw': CPLaw,
    'numberOfParams': numberOfParams,
    'initialSims': initialSims,
    'curveIndex': curveIndex,
    'projectPath': projectPath,
    'algorithm': algorithm,
    'material': material,
    'mode': mode, 
    'editLinesPH': editlinesPH,
    'editLinesDB': editLinesDB
}
sim = SIM(info)

# -------------------------------------------------------------------
#   Stage 1: Running initial simulations/Loading progress and preparing data
#   Manual variables to be changed: initial_simulations_completed, dropUpperEnd
# -------------------------------------------------------------------

print("--------------------------------")
print("Stage 1: Running initial simulations/Loading progress and preparing data")
# If you haven't run initial simulations, set it to False. Otherwise set it to True
initial_simulations_completed = True # Please change this
if initial_simulations_completed:
    initial_data = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_simulations.npy', allow_pickle=True)
    initial_data = initial_data.tolist()
    # If we already have run some few iterations and want to continue our work, we can use simulations.npy
    # initial_simulations.npy contains only the initial simulations data in a dictionary
    # simulations.npy contains all the initial simulations plus the iterations. 
    # If you want to see the content of these npy file, please use the file helpernpy.ipynb
    if path.exists(f"results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy"):
        full_data = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', allow_pickle=True)
        full_data = full_data.tolist()
        sim.simulations = full_data
        print(f"{len(initial_data)} initial simulations completed.")
        print(f"{len(full_data) - len(initial_data)} additional simulations completed.")
        print(f"Total: {len(full_data)} simulations completed.")
    else: 
        sim.simulations = initial_data
        print(f"{len(sim.simulations)} initial simulations completed.")
        print(f"No additional simulations completed.")
    allstrains = list(map(lambda x: x[0], list(initial_data.values())))
    sim.strain = np.array(allstrains).mean(axis=0)
    sim.fileNumber = len(sim.simulations)
else: 
    # If you have run initial simulations and completed preprocessing, but CSC fails at postprocessing stage, 
    # you can continue to run postprocessing by setting preprocessing_completed to True. 
    # For normal process, just set it to False
    preprocessing_completed = False # Please change this
    # If we havent run the initial simulations
    print("Running initial simulations...")
    if preprocessing_completed: 
        sim.run_initial_simulations_post()
    else:
        if method == "auto":
            sim.run_initial_simulations_auto()
        elif method == "manual":
            manualParams = np.load(f'manualParams_{material}/{CPLaw}{curveIndex}.npy', allow_pickle=True).tolist()
            tupleParams = list(map(lambda x: tuple(x), manualParams))
            sim.run_initial_simulations_manual(tupleParams)
    np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_simulations.npy', sim.simulations)
    print(f"Done. {len(sim.simulations)} simulations completed.")

exp_curve = pd.read_csv(f'targets_{material}/{CPLaw}{curveIndex}.csv')   # <--- Target curve file path.
exp_stress = exp_curve.iloc[:,0] # Getting the experimental stress
exp_strain = exp_curve.iloc[:,1] # Getting the experimental strain
# The common strain points of experimental and simulated curves will be lying between 0.002 (strain of yield stress)
# and the maximum strain value of experimental curve 
x_min, x_max = 0.002, exp_strain.max() 
# prune will be a list of True and False, which indicate which index of the strain to choose from
prune = np.logical_and(sim.strain >= x_min, sim.strain <= x_max)
# sim.strain is the average strains of the initial simulations 
# Therefore sim.strain is the same for all simulated curves. Now it is pruned
sim.strain = sim.strain[prune]
# If the error: ValueError: A value in x_new is above the interpolation range occurs,
# it is due to the the strain value of some simulated curves is higher than the last stress value
# of the interpolated strain so it lies outside the range. You can increase the dropUpperEnd number to reduce the
# range of the simulated curves so their stress can be interpolated
dropUpperEnd = 2 # Please change this
# interpolatedStrain will be the interpolating strain for all curves (experimental, initial simulation and iterated simulation)
interpolatedStrain = sim.strain[:-dropUpperEnd]
# print(interpolatedStrain)
# for (strain, stress) in sim.simulations.values():
#     print(strain[-1])
# exp_target is now the refined interpolated experimental stress values for comparison 
exp_target = interpolatedStressFunction(exp_stress, exp_strain, interpolatedStrain).reshape(-1) * convertUnit
# print(exp_target)
print("Experimental and simulated curves preparation completed")

# -------------------------------------------------------------------
#   Stage 2: Initialize Response Surface Module (MLP)
# -------------------------------------------------------------------

print("--------------------------------")
print("Stage 2: Initialize and train the RSM (MLP) with the initial data")

# -----------------------------------------
#  Initialize Response Surface Module (MLP)
# -----------------------------------------
# MLP with 1 hidden layer of 15 nodes. 
mlp = MLPRegressor(hidden_layer_sizes=[15], solver='adam', max_iter=100000, shuffle=True)
print("Fitting response surface...")
# Input layer of fitting parameters (4 for PH and 6 for DB)
X = np.array(list(sim.simulations.keys()))
# Output layer of the size of the interpolated stresses
y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
# Train the MLP
mlp.fit(X,y)
# Example of last parameter 
# print(X[-1])
# print(type(X))
# print(X.shape)
# Example of stress values at last parameter
# print(y[-1])
# print(type(y))
# print(y.shape)
print("MLP training finished")


# -----------------------------------------------------------------------
if algorithm == "GA": 
    # Set yield_stress_optimization_completed to False if you havent finished optimizing the yield stress yet
    # If you have obtained the optimized yield stress parameters already and has a saved file partial_result.npy, you can continue
    yield_stress_optimization_completed = False # Please change this 
    if yield_stress_optimization_completed:
        partialResult = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/partial_result.npy', allow_pickle=True)
        partialResult = partialResult.tolist()
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with genetic algorithm")
        fullResult = HardeningOptimizationGA(CPLaw, material, param_range_no_round, mlp, exp_target, interpolatedStrain, sim, param_range, curveIndex, algorithm, convertUnit, numberOfParams, partialResult)
    else:
        # -------------------------------------------------------------------
        #   Stage 3: Optimize the yield stress parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 3: Optimize the yield stress parameters with genetic algorithm")
        partialResult = YieldStressOptimizationGA(CPLaw, material, param_range_no_round, default_yield_value, mlp, exp_target, interpolatedStrain, sim, param_range, curveIndex, algorithm, convertUnit, numberOfParams)
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------

        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with genetic algorithm")
        fullResult = HardeningOptimizationGA(CPLaw, material, param_range_no_round, mlp, exp_target, interpolatedStrain, sim, param_range, curveIndex, algorithm, convertUnit, numberOfParams, partialResult)

# -----------------------------------------------------------------------
elif algorithm == "BA":
    # -------------------------------------------------------------------
    #   Stage 3: Optimize the yield stress parameters with BA
    # -------------------------------------------------------------------
    
    print("--------------------------------")
    print("Stage 3: Optimize the yield stress parameters with Bayesian algorithm")
    partialResult = YieldStressOptimizationBA(CPLaw, param_range_no_round, mlp, exp_target, interpolatedStrain, sim, param_range, curveIndex, algorithm, convertUnit, numberOfParams)
    
    # -------------------------------------------------------------------
    #   Stage 3: Optimize the hardening parameters with GA
    # -------------------------------------------------------------------

    print("--------------------------------")
    print("Stage 4: Optimize the hardening parameters with Bayesian algorithm")
    fullResult = HardeningOptimizationBA()

print("--------------------------------")
print("Stage 5: CP Parameter Calibration completed")
print("The final parameter solution is: ")
print(fullResult)

# python optimize.py