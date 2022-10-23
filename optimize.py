# External libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
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
CPLaw = "DB" # Please change this

# Type "GA" for genetic algorithm
# Type "BA" for Bayesian algorithm
algorithm = "GA" # Please change this

# Please choose the number of initial simulations
initialSims = 30 # Please change this

# Type the target experimental curve index (1,2,3)
curveIndex = 1 # Please change this

# The current path folder

projectPath = os.getcwd()

# Type the material name
material = "RVE_1_40_D"
# material = "512grains512"

# Type automatic or manual methods
method = "auto"
# method = "manual"

# Setting the weights of the two yield stress objective functions:  
weightsYield = {"wy1": 0.999, "wy2": 0.001}

# Setting the weights of the four hardening objective functions:  
weightsHardening = {"wh1": 0.9, "wh2": 0.025, "wh3": 0.05, "wh4": 0.025}
# Define the yield stress deviation percentage for the first stage of yield stress optimization
yieldStressDev = 0.5  # deviation for the real simulated yield stress result

# Define the hardening deviation percentage for the second stage of hardening optimization
hardeningDev = 2  # deviation for the real simulated global curve result

# If you haven't run initial simulations, set it to False. Otherwise set it to True
initial_simulations_completed = True # Please change this

if material == "RVE_1_40_D":
    param_ranges = param_ranges_RVE_1_40_D
    default_yield_values = default_yield_RVE_1_40_D 
    # These number are obtained from material.config files, where each number is the line number of the parameter minus 1
    # For example, the line of the parameter a in RVE_1_40_D in the material.config file template is 34 => 33
    # Lines of     a, h0, tau0, taucs 
    editlinesPH = [33,  34,  31,   32]
    # Lines of   dipole, islip, omega, p,  q, tausol
    editLinesDB = [66,    62,    65,  58,  59,  49]
    # This RVE is large so needs to run heavy job files
    mode = "heavy"
elif material == "512grains512":
    param_ranges = param_ranges_512grains512
    default_yield_values = default_yield_512grains512 
    # Lines of     a, h0, tau0, taucs 
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

# The default stress unit is in MPa
if CPLaw == "PH":
    numberOfParams = 4
    # In PH model, the unit is in MPa so no need to convert unit
    convertUnit = 1
    
elif CPLaw == "DB":
    numberOfParams = 6
    # In DB model, the unit is in Pa so it needs to be converted to MPa
    convertUnit = 1e-6 # In DAMASK simulation, stress unit is Pa instead of MPa in DB model

# tau0 is the main param affecting yield stress in PH model
# p, q, tausol is the main param affecting yield stress in DB model
print("\nWelcome to Crystal Plasticity Parameter Calibration")
print("\nThe configurations you have chosen: ")
print("\nMaterial under study:", material)
if CPLaw == "PH":
    law = "phenomenological law"
elif CPLaw == "DB":
    law = "dislocation-based law"
print("\nCP Law:", law)
target_curve = f"{CPLaw}{curveIndex}"
print("\nThe target curve:", target_curve)
print("\nNumber of fitting parameters in", CPLaw, "law:", numberOfParams)
print("\nRange and step of parameters: ")
print(param_range_no_round)
print("\nDefault values of hardening parameters for yield stress optimization:")
print(default_yield_value)
print("\nNumber of initial simulations:", initialSims)
print("\nChosen optimization algorithm:", algorithm)
print("\nThe optimization process is", method)
yieldStressDevPercent = f"{yieldStressDev}%"
print("\nThe yield stress deviation percentage is:", yieldStressDevPercent)
hardeningDevPercent = f"{hardeningDev}%"
print("\nThe hardening deviation percentage is", hardeningDevPercent)
print("\nThe weights wy1, wy2 of yield stress objective functions are:")
print(weightsYield)
print("\nThe weights wh1, wh2, wh3, wh4 of hardening objective functions are:")
print(weightsHardening)
print("\nThe optimization process is", method)
print("\nThe path to your main project folder is: ")
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
# If the error: ValueError: A value in x_new is above the interpolation range occurs,
# it is due to the the strain value of some simulated curves is higher than the last stress value
# of the interpolated strain so it lies outside the range. You can increase the dropUpperEnd number to reduce the
# range of the simulated curves so their stress can be interpolated
# interpolatedStrain will be the interpolating strain for all curves (experimental, initial simulation and iterated simulation)
#                                                     yield stress strain level   dropUpperENd
interpolatedStrain = calculateInterpolatingStrains(sim.strain, exp_strain, 0.002, 2) # You can change the last param dropUpperEnd
print(interpolatedStrain)
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

print("Fitting response surface method (multilayer perceptron)...")
# Input layer of fitting parameters (4 for PH and 6 for DB)
X = np.array(list(sim.simulations.keys()))
# Output layer of the size of the interpolated stresses
y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
inputSize = X.shape[1]
outputSize = y.shape[1]
print("Input layer size is:", inputSize)
print("Output layer size is:", outputSize)
#hiddenSize1 = inputSize + round((1/2) * (outputSize - inputSize))
#hiddenSize2 = inputSize + round((2/3) * (outputSize - inputSize))
#hiddenSize = [hiddenSize1, hiddenSize2]
hiddenSize = round((2/3) * inputSize + outputSize)
print("Hidden layer size is:", hiddenSize)
mlp = MLPRegressor(hidden_layer_sizes=hiddenSize, solver='adam', max_iter=100000, shuffle=True)
mlp.fit(X, y)
print("MLP training finished")

# ----------------------------------------------------------------------
# Optimization stage
# -----------------------------------------------------------------------
yieldStressOptimizeInfo = {
    "material": material,
    "CPLaw": CPLaw,
    "curveIndex": curveIndex,
    "yieldStressDev": yieldStressDev,
    "algorithm": algorithm,
    "convertUnit": convertUnit,
    "weightsYield": weightsYield,
    "numberOfParams": numberOfParams,
    "param_range": param_range,
    "param_range_no_round": param_range_no_round,
    "param_range_no_step": param_range_no_step,
    "exp_target": exp_target,
    "default_yield_value": default_yield_value,
    "interpolatedStrain": interpolatedStrain,
    "sim": sim,
    "mlp": mlp
}

hardeningOptimizeInfo = {
    "material": material,
    "CPLaw": CPLaw,
    "curveIndex": curveIndex,
    "hardeningDev": hardeningDev,
    "algorithm": algorithm,
    "weightsHardening": weightsHardening,
    "convertUnit": convertUnit,
    "numberOfParams": numberOfParams,
    "param_range": param_range,
    "param_range_no_round": param_range_no_round,
    "param_range_no_step": param_range_no_step,
    "exp_target": exp_target,
    "interpolatedStrain": interpolatedStrain,
    "sim": sim,
    "mlp": mlp
}

# Set yield_stress_optimization_completed to False if you havent finished optimizing the yield stress yet
# If you have obtained the optimized yield stress parameters already and has a saved file partial_result.npy, you can continue
yield_stress_optimization_completed = False # Please change this 

if algorithm == "GA": 
    if yield_stress_optimization_completed:
        partialResult = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/partial_result.npy', allow_pickle=True)
        partialResult = partialResult.tolist()
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with genetic algorithm")
        hardeningOptimizeInfo["partialResult"] = partialResult
        fullResult = HardeningOptimizationGA(hardeningOptimizeInfo)
    else:
        # -------------------------------------------------------------------
        #   Stage 3: Optimize the yield stress parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 3: Optimize the yield stress parameters with genetic algorithm")
        partialResult = YieldStressOptimizationGA(yieldStressOptimizeInfo)
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with genetic algorithm")
        hardeningOptimizeInfo["partialResult"] = partialResult
        fullResult = HardeningOptimizationGA(hardeningOptimizeInfo)

# -----------------------------------------------------------------------
elif algorithm == "BA":
    if yield_stress_optimization_completed:
        partialResult = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/partial_result.npy', allow_pickle=True)
        partialResult = partialResult.tolist()
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with Bayesian algorithm")
        hardeningOptimizeInfo["partialResult"] = partialResult
        fullResult = HardeningOptimizationBA(hardeningOptimizeInfo)
    else:
        # -------------------------------------------------------------------
        #   Stage 3: Optimize the yield stress parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 3: Optimize the yield stress parameters with Bayesian algorithm")
        partialResult = YieldStressOptimizationBA(yieldStressOptimizeInfo)
        # -------------------------------------------------------------------
        #   Stage 4: Optimize the hardening parameters with GA
        # -------------------------------------------------------------------
        print("--------------------------------")
        print("Stage 4: Optimize the hardening parameters with Bayesian algorithm")
        hardeningOptimizeInfo["partialResult"] = partialResult
        fullResult = HardeningOptimizationBA(hardeningOptimizeInfo)

print("--------------------------------")
print("Stage 5: CP Parameter Calibration completed")
print("The final parameter solution is: ")
print(fullResult)

# python optimize.py