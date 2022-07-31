# External libraries
import pandas as pd
import numpy as np
import pygad
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
# Our classes
from classes.SIM_PH import *
from classes.SIM_DB import *
from classes.preprocessing import *
from classes.fitness import *
from classes.param_range import *
from os import path
import time

# -------------------------------------------------------------------
#   Choose the CP model, the optimization algorithm, number of initial simulations,
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
# projectPath = "/scratch/project_2004956/Binh/DB1BayesLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/DB1GeneticLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/DB2BayesLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/DB2BayesLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/DB3GeneticLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/DB3GeneticLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/PH1BayesLargeRVE/"
projectPath = "/scratch/project_2004956/Binh/PH1GeneticLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/PH2BayesLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/PH2GeneticLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/PH3BayesLargeRVE/"
# projectPath = "/scratch/project_2004956/Binh/PH3GeneticLargeRVE/"

# Type the material name
material = "RVE_1_40_D"
# material = "512grains512"

# -------------------------------------------------------------------
#   Run initial simulations (used to initialize optimization process).
# -------------------------------------------------------------------

if CPLaw == "PH":
    param_range = param_ranges_PH[curveIndex - 1]
    param_range_no_round = param_ranges_no_round_PH[curveIndex - 1]
    param_range_no_step = param_ranges_no_step_PH[curveIndex - 1]
    numberOfParams = 4
    # Initialize the SIM object.
    info = {
        'param_range': param_range,
        'numberOfParams': numberOfParams,
        'initialSims': initialSims,
        'curveIndex': curveIndex,
        'projectPath': projectPath,
        'algorithm': algorithm,
        'material': material
    }
    sim = SIM_PH(info)

elif CPLaw == "DB":
    param_range = param_ranges_DB[curveIndex - 1]
    param_range_no_round = param_ranges_no_round_DB[curveIndex - 1]
    param_range_no_step = param_ranges_no_step_DB[curveIndex - 1]
    numberOfParams = 6
    # Initialize the SIM object.
    info = {
        'param_range': param_range,
        'numberOfParams': numberOfParams,
        'initialSims': initialSims,
        'curveIndex': curveIndex,
        'projectPath': projectPath,
        'algorithm': algorithm,
        'material': material
    }
    sim = SIM_DB(info)

# If you haven't run initial simulations, set it to False. Otherwise set it to True
initial_simulations_completed = True # Please change this
if initial_simulations_completed:
    initial_data = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_simulations.npy', allow_pickle=True)
    initial_data = initial_data.tolist()
    # If we already have run some few iterations and want to continue our work
    # initial_simulations.npy contains only the initial simulations data in a dictionary
    # simulations.npy contains all the initial simulations plus the iterations. 
    # If you want to see the content of these npy file, please use the file 
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
    sim.filename = len(sim.simulations)
else: 
    # If we havent run the initial simulations
    print("Running Initial Simulations...")
    sim.run_initial_simulations()
    np.save('results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_simulations.npy', sim.simulations)
    print(f"Done. {len(sim.simulations)} simulations completed.")

# -------------------------------------
#   Set up experimental curve (target)
# -------------------------------------

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
exp_target = interpolatedFunction(sim.strain).reshape(-1)

# -----------------------------------------
#  Initialize Response Surface Module (MLP)
# -----------------------------------------
# MLP with 1 hidden layer of 15 nodes. 
mlp = MLPRegressor(hidden_layer_sizes=[15], solver='adam', max_iter=100000, shuffle=True)
print("Fitting response surface...")
# Input layer of four parameters
X = np.array(list(sim.simulations.keys()))
# Input layer of the size of the pruned stresses
y = np.array([stress[prune] for (_, stress) in sim.simulations.values()])
# Train the MLP
mlp.fit(X,y)

# Starting initialization of algorithms
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

    # -------------------------------
    #      Helper Functions
    # -------------------------------

    def output_results(ga_instance):
        # Returning the details of the best solution in a dictionary.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        best_solution_generation = ga_instance.best_solution_generation
        fitness = 1/solution_fitness
        values = (solution, solution_fitness, solution_idx, best_solution_generation, fitness)
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
    #      Optimization Loop
    # -------------------------------

    print("Optimizing using GA...")
    ga_instance.run()
    results = output_results(ga_instance)
    while tuple(results['solution']) in sim.simulations.keys():
        print("Parameters already probed. Algorithm need to run again to obtain new parameters")
        ga_instance.run()
        results = output_results(ga_instance)

    print_results(results)

    # Wait a moment so that you can check the parameters predicted by the algorithm
    # time.sleep(20)

    # Iterative optimization.
    while not insideFivePercentStd(exp_target, y[-1]) or not closeYield(exp_target, y[-1]):
        sim.run_single_test(tuple(results['solution']))
        np.save(f'results/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', sim.simulations)
        y = np.array([stress[prune] for (_, stress) in sim.simulations.values()])
        X = np.array(list(sim.simulations.keys()))
        mlp.fit(X,y)
        loss = mean_squared_error(y[-1], exp_target)
        print(f"MSE LOSS = {loss}")
        print("--------------------------------")
        ga_instance.run()
        results = output_results(ga_instance)
        while tuple(results['solution']) in sim.simulations.keys():
            print("Parameters already probed. Algorithm need to run again to obtain new parameters")
            ga_instance.run()
            results = output_results(ga_instance)
        results = output_results(ga_instance)
        print_results(results)
        # Wait a moment so that you can check the parameters predicted by the algorithm
        time.sleep(60)

elif algorithm == "BA":
    # -------------------------------
    #      Initialize BA
    # -------------------------------

    print("Optimizing using BA...")

print("Optimization Complete")
print("--------------------------------")
print("Final Parameters: ", results['solution'])
print(f"MSE of the final solution : {loss}")

# python optimize.py