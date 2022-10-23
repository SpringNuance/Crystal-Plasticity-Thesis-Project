from modules.fitness import *
from modules.helper import *
import pygad
import time
from sklearn.metrics import mean_squared_error
from math import *

def output_resultsPartialGA(ga_instance, param_range, default_yield_value, CPLaw):
    # Returning the details of the best solution in a dictionary.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    best_solution_generation = ga_instance.best_solution_generation
    fitness = 1/solution_fitness
    if CPLaw == "PH":
        solution_dict = {
            'a': default_yield_value['a'],
            'h0': default_yield_value['h0'],
            'tau0': solution[0],
            'taucs': default_yield_value['taucs']
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': default_yield_value['dipole'],
            'islip': default_yield_value['islip'],
            'omega': default_yield_value['omega'],
            'p': solution[0],
            'q': solution[1], 
            'tausol': solution[2]
        }
 
    solution_dict_round = round_params(solution_dict, param_range)
    solution_list = list(solution_dict_round.values())
    values = (solution_list, solution_dict_round, solution_fitness, solution_idx, best_solution_generation, fitness)
    keys = ("solution", "solution_dict", "solution_fitness", "solution_idx", "best_solution_generation", "fitness")
    output = dict(zip(keys, values))
    return output

def output_resultsFullGA(ga_instance, param_range, partialResult, CPLaw):
    # Returning the details of the best solution in a dictionary.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    best_solution_generation = ga_instance.best_solution_generation
    fitness = 1/solution_fitness
    if CPLaw == "PH":
        solution_dict = {
            'a': solution[0],
            'h0': solution[1],
            'tau0': partialResult['tau0'],
            'taucs': solution[2]
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': solution[0],
            'islip': solution[1],
            'omega': solution[2],
            'p': partialResult['p'],
            'q': partialResult['q'], 
            'tausol': partialResult['tausol'],
        }
    solution_dict_round = round_params(solution_dict, param_range)
    solution_list = list(solution_dict_round.values())
    values = (solution_list, solution_dict_round, solution_fitness, solution_idx, best_solution_generation, fitness)
    keys = ("solution", "solution_dict", "solution_fitness", "solution_idx", "best_solution_generation", "fitness")
    output = dict(zip(keys, values))
    return output

def print_resultsPartialGA(results):
    print(f"Parameters of the best partial solution : {results['solution_dict']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Index of the best solution : {results['solution_idx']}")
    print(f"Fitness given by the MLP estimate: {results['fitness']}")

def print_resultsFullGA(results):
    print(f"Parameters of the best full solution : {results['solution_dict']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Index of the best solution : {results['solution_idx']}")
    print(f"Fitness given by the MLP estimate: {results['fitness']}")

last_fitness = 0
keep_parents = 1

def YieldStressOptimizationGA(yieldStressOptimizeInfo):
    material = yieldStressOptimizeInfo["material"]
    CPLaw = yieldStressOptimizeInfo["CPLaw"]
    curveIndex = yieldStressOptimizeInfo["curveIndex"] 
    yieldStressDev = yieldStressOptimizeInfo["yieldStressDev"]
    algorithm = yieldStressOptimizeInfo["algorithm"] 
    weightsYield = yieldStressOptimizeInfo["weightsYield"]
    convertUnit = yieldStressOptimizeInfo["convertUnit"] 
    numberOfParams = yieldStressOptimizeInfo["numberOfParams"] 
    param_range = yieldStressOptimizeInfo["param_range"] 
    param_range_no_round = yieldStressOptimizeInfo["param_range_no_round"] 
    exp_target = yieldStressOptimizeInfo["exp_target"] 
    default_yield_value = yieldStressOptimizeInfo["default_yield_value"] 
    interpolatedStrain = yieldStressOptimizeInfo["interpolatedStrain"] 
    sim = yieldStressOptimizeInfo["sim"] 
    mlp = yieldStressOptimizeInfo["mlp"] 
    wy1 = weightsYield["wy1"]
    wy2 = weightsYield["wy2"]



    # -------------------------------
    #      Initialize GA
    # -------------------------------
    if CPLaw == "PH":
        gene_space = [param_range_no_round['tau0']]
        numberOfYieldStressParams = 1
    elif CPLaw == "DB":
        gene_space = [param_range_no_round['p'], param_range_no_round['q'], param_range_no_round['tausol']]
        numberOfYieldStressParams = 3
    num_genes = numberOfYieldStressParams
    
    def fitnessYieldGA(solution, solution_idx):
        if CPLaw == "PH":
            partialSolution = np.array([default_yield_value['a'], default_yield_value['h0'], solution[0], default_yield_value['taucs']])
        elif CPLaw == "DB":
            partialSolution = np.array([default_yield_value['dipole'], default_yield_value['islip'], default_yield_value['omega'], solution[0], solution[1], solution[2]])
        predicted_sim_stress = mlp.predict(partialSolution.reshape((1, numberOfParams))).reshape(-1)
        chromosomefit = fitness_yield(exp_target, predicted_sim_stress, interpolatedStrain, wy1, wy2)
        fitnessScore = 1/chromosomefit
        return fitnessScore
    
    def on_generation(ga_instance):
        global last_fitness
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
        last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    ga_instance = pygad.GA(num_generations=100, # Number of generations.
                        num_parents_mating=500, # Number of solutions to be selected as parents in the mating pool.
                        sol_per_pop=1000, # Number of solutions in the population.
                        num_genes=num_genes,
                        fitness_func=fitnessYieldGA,
                        on_generation=on_generation,
                        gene_space=gene_space,
                        crossover_type="single_point",
                        mutation_type="random",
                        mutation_num_genes=1)

    print("The experimental yield stress is: ", exp_target[0], "MPa")
    rangeSimYield = (exp_target[0]* (1 - yieldStressDev * 0.01), exp_target[0] * (1 + yieldStressDev * 0.01)) 
    print("The simulated yield stress should lie in the range of", rangeSimYield, "MPa")
    print("Maximum deviation:", exp_target[0] * yieldStressDev * 0.01, "MPa")
    print("#### Iteration", sim.fileNumber, "####")
    y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
    # If you want to find the best result from the initial random initial sims, you can set to true. It is likely that
    # one of the initial sims have yield stress close to the experimental yield stress so you can save time optimizing the yield stress
    bestResultFromInitialSimsLucky = False
    if bestResultFromInitialSimsLucky:
        zipParamsStress = list(zip(list(sim.simulations.keys()), y))
        sortedClosestYieldStress = list(sorted(zipParamsStress, key=lambda pairs: fitness_yield(exp_target, pairs[1], interpolatedStrain, wy1, wy2), reverse=True))
        y = np.array(list(map(lambda x: x[1], sortedClosestYieldStress)))
        partialResult = sortedClosestYieldStress[-1][0]
        partialResult = tupleOrListToDict(partialResult, CPLaw)
    else:
        partialResult = list(sim.simulations.keys())[-1]
        partialResult = tupleOrListToDict(partialResult, CPLaw)
    print("The initial candidate partial result: ")
    print(partialResult)
    print("The initial candidate simulated yield stress: ")
    print(y[-1][0])
    # Iterative optimization.
    while not insideYieldStressDev(exp_target, y[-1], yieldStressDev):
        print("#### Iteration", sim.fileNumber + 1, "####")
        ga_instance.run()
        partialResults = output_resultsPartialGA(ga_instance, param_range, default_yield_value, CPLaw)
        while tuple(partialResults['solution']) in sim.simulations.keys():
            print("The predicted solution is:")
            print(partialResults["solution_dict"])
            print("Parameters already probed. Algorithm needs to run again to obtain new parameters")
            ga_instance.run()
            partialResults = output_resultsPartialGA(ga_instance, param_range, default_yield_value, CPLaw)
        print_resultsPartialGA(partialResults)
        # Wait a moment so that you can check the parameters predicted by the algorithm 
        time.sleep(20)
        partialResult = partialResults['solution_dict']
        sim.run_single_test(tuple(partialResults['solution']))
        np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', sim.simulations)
        X = np.array(list(sim.simulations.keys()))
        y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
        mlp.fit(X, y)
        print("The simulated yield stress:", y[-1][0],"MPa")
    print("--------------------------------")
    print("Yield stress parameters optimization completed")
    partialResult = defaultParams(partialResult, CPLaw, default_yield_value)
    print("The partial parameter solution is: ")
    print(partialResult)
    print("Succeeded iteration:", sim.fileNumber)
    np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/partial_result.npy', partialResult)
    return partialResult

def HardeningOptimizationGA(hardeningOptimizeInfo):
    # -------------------------------
    #      Initialize GA
    # -------------------------------
    material = hardeningOptimizeInfo["material"]
    CPLaw = hardeningOptimizeInfo["CPLaw"]
    curveIndex = hardeningOptimizeInfo["curveIndex"] 
    hardeningDev = hardeningOptimizeInfo["hardeningDev"] 
    algorithm = hardeningOptimizeInfo["algorithm"] 
    weightsHardening = hardeningOptimizeInfo["weightsHardening"]
    convertUnit = hardeningOptimizeInfo["convertUnit"] 
    numberOfParams = hardeningOptimizeInfo["numberOfParams"] 
    param_range = hardeningOptimizeInfo["param_range"] 
    param_range_no_round = hardeningOptimizeInfo["param_range_no_round"] 
    exp_target = hardeningOptimizeInfo["exp_target"] 
    interpolatedStrain = hardeningOptimizeInfo["interpolatedStrain"] 
    sim = hardeningOptimizeInfo["sim"] 
    mlp = hardeningOptimizeInfo["mlp"] 
    partialResult = hardeningOptimizeInfo["partialResult"] 
    wh1 = weightsHardening["wh1"]
    wh2 = weightsHardening["wh2"]
    wh3 = weightsHardening["wh3"]
    wh4 = weightsHardening["wh4"]

    if CPLaw == "PH":
        gene_space = [param_range_no_round['a'], param_range_no_round['h0'], param_range_no_round['taucs']]
        numberOfHardeningParams = 3
    elif CPLaw == "DB":
        gene_space = [param_range_no_round['dipole'], param_range_no_round['islip'], param_range_no_round['omega']]
        numberOfHardeningParams = 3
    num_genes = numberOfHardeningParams
    
    def fitnessHardeningGA(solution, solution_idx):
        if CPLaw == "PH":
            fullSolution = np.array([solution[0], solution[1], partialResult['tau0'], solution[2]])
        elif CPLaw == "DB":
            fullSolution = np.array([solution[0], solution[1], solution[2], partialResult['p'], partialResult['q'], partialResult['tausol']])
        predicted_sim_stress = mlp.predict(fullSolution.reshape((1, numberOfParams))).reshape(-1)
        chromosomefit = fitness_hardening(exp_target, predicted_sim_stress, interpolatedStrain, wh1, wh2, wh3, wh4)
        fitnessScore = 1/chromosomefit
        return fitnessScore

    def on_generation(ga_instance):
        global last_fitness
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
        last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    ga_instance = pygad.GA(num_generations=100, # Number of generations.
                        num_parents_mating=500, # Number of solutions to be selected as parents in the mating pool.
                        sol_per_pop=1000, # Number of solutions in the population.
                        num_genes=num_genes,
                        fitness_func=fitnessHardeningGA,
                        on_generation=on_generation,
                        gene_space=gene_space,
                        crossover_type="single_point",
                        mutation_type="random",
                        mutation_num_genes=1)
    
    
    print("The partial result: ")
    print(partialResult)
    y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
    fullResult = list(sim.simulations.keys())[-1]
    print("The initial candidate full result: ")
    print(fullResult)
    # Iterative optimization.
    while not insideHardeningDev(exp_target, y[-1], hardeningDev):
        print("#### Iteration", sim.fileNumber + 1, "####")
        ga_instance.run()
        fullResults = output_resultsFullGA(ga_instance, param_range, partialResult, CPLaw)
        while tuple(fullResults['solution']) in sim.simulations.keys():
            print("The predicted solution is:")
            print(fullResults["solution_dict"])
            print("Parameters already probed. Algorithm needs to run again to obtain new parameters")
            ga_instance.run()
            fullResults = output_resultsFullGA(ga_instance, param_range, partialResult, CPLaw)
        print_resultsFullGA(fullResults)
        # Wait a moment so that you can check the parameters predicted by the algorithm 
        time.sleep(20)
        fullResult = fullResults['solution_dict']
        sim.run_single_test(tuple(fullResults['solution']))
        np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/simulations.npy', sim.simulations)
        X = np.array(list(sim.simulations.keys()))
        y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
        mlp.fit(X, y)
        loss = sqrt(mean_squared_error(y[-1], exp_target))
        print(f"RMSE LOSS = {loss}")
    print("--------------------------------")
    print("Hardening parameters optimization completed")
    print("The full parameter solution is: ")
    print(fullResult)
    print("Succeeded iteration:", sim.fileNumber)
    np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/full_result.npy', fullResult)
    return fullResult

