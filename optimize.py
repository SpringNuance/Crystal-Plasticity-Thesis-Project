# External libraries
import pandas as pd
import numpy as np
import pygad
import bayes_opt 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
# Our classes
from classes.SIM import *
from classes.preprocessing import *
from classes.fitness import *
from classes.param_ranges import *
from classes.helper import *
from os import path
import time

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
projectPath = "/scratch/project_2004956/Binh/CrystalPlasticityProject"

# Type the material name
material = "RVE_1_40_D"
# material = "512grains512"

# Type automatic or manual methods
method = "auto"
# method = "manual"

if material == "RVE_1_40_D":
    param_ranges = param_ranges_RVE_1_40_D
    # These number are obtained from material.config files, where each number is the line number of the parameter minus 1
    # For example, the line of the parameter alpha in RVE_1_40_D in the material.config file template is 34 => 33
    # Lines of   alpha, h0, tau0, taucs 
    editlinesPH = [33,  34,  31,   32]
    # Lines of   dipole, islip, omega, p,  q, tausol
    editLinesDB = [66,    62,    65,  58,  59,  49]
    # This RVE is large so needs more nodes to run
    nodes = 8
elif material == "512grains512":
    param_ranges = param_ranges_512grains512
    # Lines of   alpha, h0, tau0, taucs 
    editlinesPH = [54,  46,  36,   37]
    # Lines of   dipole, islip, omega, p,  q, tausol
    editLinesDB = [48,    44,    47,  40,  41,  31]
    # This RVE is small so needs fewer nodes to run
    nodes = 4

param_range = param_ranges[CPLaw][curveIndex - 1] # param_range is used for SIM object
param_range_no_round = param_range_no_round_func(param_range) # param_range_no_round is used to be fed to GA
param_range_no_step = param_range_no_step_func(param_range_no_round) # param_range_no_step is used to be fed to BA
print("param_range is:")
print(param_range)
print("param_no_round is:")
print(param_range_no_round)
print("param_range_no_step is:")
print(param_range_no_step)

if CPLaw == "PH":
    numberOfParams = 4
    convertUnit = 1
elif CPLaw == "DB":
    numberOfParams = 6
    convertUnit = 1e-6 # In DAMASK simulation, stress unit is Pa instead of MPa

print("Welcome to Crystal Plasticity Parameter Calibration")
print("The configurations you have chosen: ")
print("CP Law:", CPLaw)
target_curve = f"{CPLaw}{curveIndex}"
print("The target curve:", target_curve)
print("Number of fitting parameters in", CPLaw, "law:", numberOfParams)
print("Range and step of parameters: ")
print(param_range_no_round)
print("Number of initial simulations:", initialSims)
print("Chosen optimization algorithm:", algorithm)
print("Material under study:",material)
print("The optimization process is", method)
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
    'nodes': nodes, 
    'editLinesPH': editlinesPH,
    'editLinesDB': editLinesDB
}
sim = SIM(info)

# -------------------------------------------------------------------
#   Stage 1: Running initial simulations/Loading progress and preparing data
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
    if path.exists("results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy"):
        full_data = np.load('results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', allow_pickle=True)
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
    # If we havent run the initial simulations
    print("Running initial simulations...")
    if method == "auto":
        sim.run_initial_simulations_auto()
    elif method == "manual":
        manualParams = np.load(f'manualParams/{CPLaw}{curveIndex}.npy', allow_pickle=True).tolist()
        tupleParams = list(map(lambda x: tuple(x), manualParams))
        sim.run_initial_simulations_manual(tupleParams)
    np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_simulations.npy', sim.simulations)
    print(f"Done. {len(sim.simulations)} simulations completed.")

exp_curve = pd.read_csv(f'targets_{material}/{CPLaw}{curveIndex}.csv')   # <--- Target curve file path.
exp_stress = exp_curve.iloc[:,0] # Getting the experimental stress
exp_strain = exp_curve.iloc[:,1] # Getting the experimental strain
interpolatedFunction = interp1d(exp_strain, exp_stress) # interpolated function fits the experimental data
# The common strain points of experimental and simulated curves will be lying between 0.002 (strain of yield stress)
# and the maximum strain value of experimental curve 
x_min, x_max = 0.002, exp_strain.max() 
# prune will be a list of True and False, which indicate which index of the strain to choose from
prune = np.logical_and(sim.strain > x_min, sim.strain < x_max)
# sim.strain is the average strains of the initial simulations 
# Therefore sim.strain is the same for all simulated curves. Now it is pruned
sim.strain = sim.strain[prune]
# exp_target is now the refined experimental stress values for comparison such as fitness and MSE, 
# after being pruned and interpolated at the pruned simulated stress values
exp_target = interpolatedFunction(sim.strain).reshape(-1) * convertUnit

print("Experimental and simulated curves preparation completed")

# -------------------------------------------------------------------
#   Stage 2: Initialize Response Surface Module (MLP)
# -------------------------------------------------------------------

print("--------------------------------")
print("Stage 2: Initialize and train the RSM (MLP)")

# -----------------------------------------
#  Initialize Response Surface Module (MLP)
# -----------------------------------------
# MLP with 1 hidden layer of 15 nodes. 
mlp = MLPRegressor(hidden_layer_sizes=[15], solver='adam', max_iter=100000, shuffle=True)
print("Fitting response surface...")
# Input layer of four parameters
X = np.array(list(sim.simulations.keys()))
# Input layer of the size of the pruned stresses
y = np.array([stress[prune] * convertUnit for (_, stress) in sim.simulations.values()])
# Train the MLP
mlp.fit(X,y)
print("MLP training finished")

# -------------------------------------------------------------------
#   Stage 3: Initialize and run optimization algorithm
# -------------------------------------------------------------------

print("--------------------------------")
print("Stage 3: Initialize and run optimization algorithm")
if algorithm == "GA":
    # -------------------------------
    #      Initialize GA
    # -------------------------------

    # Initialize fitness function
    def fitness(solution, solution_idx):
        predicted_sim_stress = mlp.predict(solution.reshape((1,numberOfParams))).reshape(-1)
        chromosomefit = chromosomefitness(exp_target, predicted_sim_stress, sim.strain, w1, w2, w3, w4)
        fitnessScore = 1/chromosomefit
        return fitnessScore

    # Initialize GA Optimizer
    num_generations = 100 # Number of generations.
    num_parents_mating = 500 # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 1000 # Number of solutions in the population.
    if CPLaw == "PH":
        gene_space = [param_range_no_round['alpha'], param_range_no_round['h0'], param_range_no_round['tau'], param_range_no_round['taucs']]
    elif CPLaw == "DB":
        gene_space = [param_range_no_round['dipole'], param_range_no_round['islip'], param_range_no_round['omega'], param_range_no_round['p'], param_range_no_round['q'], param_range_no_round['tausol']]
    num_genes = numberOfParams
    last_fitness = 0
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 25
    
    def on_generation(ga_instance):
        global last_fitness
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
        last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        fitness_func=fitness,
                        on_generation=on_generation,
                        gene_space=gene_space,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes)
    
    # Helper functions
    def output_results(ga_instance):
        # Returning the details of the best solution in a dictionary.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        best_solution_generation = ga_instance.best_solution_generation
        fitness = 1/solution_fitness
        if CPLaw == "PH":
            solution_dict = {
                'alpha': solution[0],
                'h0': solution[1],
                'tau0': solution[2],
                'taucs': solution[3]
            }
        elif CPLaw == "DB":
            solution_dict = {
                'dipole': solution[0],
                'islip': solution[1],
                'omega': solution[2],
                'p': solution[3],
                'q': solution[4], 
                'tausol': solution[5]
            }
        values = (solution_dict, solution_fitness, solution_idx, best_solution_generation, fitness)
        keys = ("solution", "solution_fitness", "solution_idx", "best_solution_generation", "fitness")
        output = dict(zip(keys, values))
        return output
    
    def print_results(results):
        print(f"Parameters of the best solution : {results['solution']}")
        print(f"Fitness value of the best solution = {results['solution_fitness']}")
        print(f"Index of the best solution : {results['solution_idx']}")
        print(f"Fitness given by the MLP estimate: {results['fitness']}")
        print(f"Best fitness value reached after {results['best_solution_generation']} generations.")
    
    # -------------------------------
    #      End of GA
    # -------------------------------

elif algorithm == "BA":
    # -------------------------------
    #      Initialize BA
    # -------------------------------

    # Initialize surrogate function
    if CPLaw == "PH":
        def surrogate(alpha, h0, tau, taucs):
            params = {
                'alpha': alpha,
                'h0': h0,
                'tau': tau,
                'taucs': taucs
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            alpha, h0, tau, taucs = round_params(params, param_range)
            solution = np.array([alpha, h0, tau, taucs])
            predicted_sim_stress = mlp.predict(solution.reshape((1, numberOfParams))).reshape(-1)
            chromosomefit = chromosomefitness(exp_target, predicted_sim_stress, sim.strain, w1, w2, w3, w4)
            fitnessScore = 1/chromosomefit
            return fitnessScore
    elif CPLaw == "DB":
        def surrogate(dipole, islip, omega, p, q, tausol):
            params = {
                'dipole': dipole,
                'islip': islip,
                'omega': omega,
                'p': p,
                'q': q, 
                'tausol': tausol
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            dipole, islip, omega, p, q, tausol = round_params(params, param_range)
            solution = np.array([dipole, islip, omega, p, q, tausol])
            predicted_sim_stress = mlp.predict(solution.reshape((1, numberOfParams))).reshape(-1)
            chromosomefit = chromosomefitness(exp_target, predicted_sim_stress, sim.strain, w1, w2, w3, w4)
            fitnessScore = 1/chromosomefit
            return fitnessScore

    # Initialize BA Optimizer
    ba_instance = bayes_opt.BayesianOptimization(f = surrogate,
                                    pbounds = param_range_no_step, verbose = 2,
                                    random_state = 4)

    def optimize_process(X, y):    
        yFitness = list(map(lambda sim_stress: 1/chromosomefitness(exp_target, sim_stress.reshape(-1), sim.strain, w1, w2, w3, w4), y))
        pairs = list(zip(X, yFitness))

        for pair in pairs:
            ba_instance.register(
                params=pair[0],
                target=pair[1],
            )
        
        # There are two ways of using BA: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # TO use automatic way, comment out sequential way, from iterations = ... until after the loop
        # Sequential way  
        '''
        iterations = 100
        utility = bayes_opt.UtilityFunction(kind="ucb", kappa=2.5, xi = 1)
        for _ in range(iterations):
            next_point = ba_instance.suggest(utility)
            target = surrogate(**next_point)
            ba_instance.register(params=next_point, target=target)
        '''
        # Automatic way
        init_points = 15
        iterations = 15
        for i in range(10):
            ba_instance.maximize(
                init_points = init_points, 
                n_iter = iterations,    
                # What follows are GP regressor parameters
                alpha=1,
                n_restarts_optimizer=5)
            ba_instance.set_gp_params(normalize_y=True)    
        results = output_results(ba_instance)
        return results

    # Helper functions
    def output_results(ba_instance):
        # Returning the details of the best solution in a dictionary.
        solution = round_params(ba_instance.max["params"])
        solution_fitness = ba_instance.max["target"]
        fitness = 1/solution_fitness
        values = (solution, solution_fitness, fitness)
        keys = ("solution", "solution_fitness", "fitness")
        output = dict(zip(keys, values))
        return output

    def print_results(results):
        print(f"Parameters of the best solution : {results['solution']}")
        print(f"Fitness value of the best solution = {results['solution_fitness']}")
        print(f"Fitness given by the MLP estimate: {results['fitness']}")
    
    # -------------------------------
    #      End of BA
    # -------------------------------

if algorithm == "GA":
    print("Optimizing using GA...")
    ga_instance.run()
    results = output_results(ga_instance)
    while tuple(results['solution']) in sim.simulations.keys():
        print("Parameters already probed. Algorithm need to run again to obtain new parameters")
        ga_instance.run()
        results = output_results(ga_instance)
    print_results(results)
elif algorithm == "BA":
    print("Optimizing using BA...")
    results = optimize_process(X, y)
    while tuple(results['solution']) in sim.simulations.keys():
        print("Parameters already probed. Algorithm need to run again to obtain new parameters")
        results = optimize_process(X, y)
    print_results(results)

# Wait a moment so that you can check the parameters predicted by the algorithm
time.sleep(10)

# -------------------------------------------------------------------
#   Stage 4: Running iterative optimization loop
# -------------------------------------------------------------------

print("--------------------------------")
print("Stage 4: Running iterative optimization loop")

# Iterative optimization.
while not insideFivePercentStd(exp_target, y[-1]) or not closeYield(exp_target, y[-1]):
    sim.run_single_test(tuple(results['solution']))
    np.save(f'results/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', sim.simulations)
    y = np.array([stress[prune] * convertUnit for (_, stress) in sim.simulations.values()])
    X = np.array(list(sim.simulations.keys()))
    mlp.fit(X,y)
    loss = mean_squared_error(y[-1], exp_target)
    print(f"MSE LOSS = {loss}")
    print("--------------------------------")
    if algorithm == "GA":
        print("Optimizing using GA...")
        ga_instance.run()
        results = output_results(ga_instance)
        while tuple(results['solution']) in sim.simulations.keys():
            print("Parameters already probed. Algorithm need to run again to obtain new parameters")
            ga_instance.run()
            results = output_results(ga_instance)
        print_results(results)
    elif algorithm == "BA":
        print("Optimizing using BA...")
        results = optimize_process(X, y)
        while tuple(results['solution']) in sim.simulations.keys():
            print("Parameters already probed. Algorithm need to run again to obtain new parameters")
            results = optimize_process(X, y)
        print_results(results)
    # Wait a moment so that you can check the parameters predicted by the algorithm
    time.sleep(60)

print("--------------------------------")
print("Stage 5: CP Parameter Calibration completed")
print("Final parameter solution: ", results['solution'])
print(f"MSE of the final solution : {loss}")

# python optimize.py