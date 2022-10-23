from modules.fitness import *
from modules.helper import *
import bayes_opt
import time
from sklearn.metrics import mean_squared_error
from math import *
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def output_resultsPartialBA(ba_instance, param_range, default_yield_value, CPLaw):
    # Returning the details of the best solution in a dictionary.
    solution_dict_original = ba_instance.max["params"]
    solution_fitness = ba_instance.max["target"]
    fitness = 1/solution_fitness
    if CPLaw == "PH":
        solution_dict = {
            'a': default_yield_value['a'],
            'h0': default_yield_value['h0'],
            'tau0': solution_dict_original['tau0'] * (10 ** - param_range["tau0"]["round"]),
            'taucs': default_yield_value['taucs']
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': default_yield_value['dipole'],
            'islip': default_yield_value['islip'],
            'omega': default_yield_value['omega'],
            'p': solution_dict_original["p"] * (10 ** - param_range["p"]["round"]),
            'q': solution_dict_original["q"] * (10 ** - param_range["q"]["round"]), 
            'tausol': solution_dict_original["tausol"] * (10 ** - param_range["tausol"]["round"])
        }
    solution_dict = round_params(solution_dict, param_range)
    solution = list(solution_dict.values())
    values = (solution, solution_dict, solution_fitness, fitness)
    keys = ("solution", "solution_dict", "solution_fitness", "fitness")
    output = dict(zip(keys, values))
    return output

def output_resultsFullBA(ba_instance, param_range, partialResult, CPLaw):
    # Returning the details of the best solution in a dictionary.
    solution_dict_original = ba_instance.max["params"]
    solution_fitness = ba_instance.max["target"]
    fitness = 1/solution_fitness
    if CPLaw == "PH":
        solution_dict = {
            'a': solution_dict_original['a'] * (10 ** - param_range["a"]["round"]),
            'h0': solution_dict_original['h0'] * (10 ** - param_range["h0"]["round"]),
            'tau0': partialResult['tau0'],
            'taucs': solution_dict_original['taucs'] * (10 ** - param_range["taucs"]["round"])
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': solution_dict_original['dipole'] * (10 ** - param_range["dipole"]["round"]),
            'islip': solution_dict_original['islip'] * (10 ** - param_range["islip"]["round"]),
            'omega': solution_dict_original['omega'] * (10 ** - param_range["omega"]["round"]),
            'p': partialResult["p"],
            'q': partialResult["q"], 
            'tausol': partialResult["tausol"]
        }
    solution_dict = round_params(solution_dict, param_range)
    solution = list(solution_dict.values())
    values = (solution, solution_dict, solution_fitness, fitness)
    keys = ("solution", "solution_dict", "solution_fitness", "fitness")
    output = dict(zip(keys, values))
    return output

def print_resultsPartialBA(results):
    print(f"Parameters of the best partial solution : {results['solution_dict']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Fitness given by the MLP estimate: {results['fitness']}")

def print_resultsFullBA(results):
    print(f"Parameters of the best full solution : {results['solution_dict']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Fitness given by the MLP estimate: {results['fitness']}")

def multiply(tupleRange, multiplier):
    return tuple(int(i * multiplier) for i in tupleRange)

def YieldStressOptimizationBA(yieldStressOptimizeInfo):
    material = yieldStressOptimizeInfo["material"]
    CPLaw = yieldStressOptimizeInfo["CPLaw"]
    curveIndex = yieldStressOptimizeInfo["curveIndex"] 
    yieldStressDev = yieldStressOptimizeInfo["yieldStressDev"]
    algorithm = yieldStressOptimizeInfo["algorithm"] 
    weightsYield = yieldStressOptimizeInfo["weightsYield"]
    convertUnit = yieldStressOptimizeInfo["convertUnit"] 
    numberOfParams = yieldStressOptimizeInfo["numberOfParams"] 
    param_range = yieldStressOptimizeInfo["param_range"] 
    param_range_no_step = yieldStressOptimizeInfo["param_range_no_step"] 
    exp_target = yieldStressOptimizeInfo["exp_target"] 
    default_yield_value = yieldStressOptimizeInfo["default_yield_value"] 
    interpolatedStrain = yieldStressOptimizeInfo["interpolatedStrain"] 
    sim = yieldStressOptimizeInfo["sim"] 
    mlp = yieldStressOptimizeInfo["mlp"] 
    wy1 = weightsYield["wy1"]
    wy2 = weightsYield["wy2"]
    
    # -------------------------------
    #      Initialize BA
    # -------------------------------
    if CPLaw == "PH":
        pbounds = {
            "tau0": multiply(param_range_no_step['tau0'], 10 ** param_range["tau0"]["round"])
        }
    elif CPLaw == "DB":
        pbounds = {
            "p": multiply(param_range_no_step['p'], 10 ** param_range["p"]["round"]), 
            "q": multiply(param_range_no_step['q'], 10 ** param_range["q"]["round"]), 
            "tausol": multiply(param_range_no_step['tausol'], 10 ** param_range["tausol"]["round"])
        }
    print(pbounds)

    # Initialize surrogate function
    if CPLaw == "PH":
        def surrogateYieldBA(tau0):
            params = {
                'a': default_yield_value["a"],
                'h0': default_yield_value["h0"],
                'tau0': tau0 * (10 ** - param_range["tau0"]["round"]),
                'taucs': default_yield_value["taucs"]
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            candidate_dict_round = round_params(params, param_range)
            solution = np.array(list(candidate_dict_round.values()))
            predicted_sim_stress = mlp.predict(solution.reshape(1, numberOfParams)).reshape(-1)
            candidateScore = fitness_yield(exp_target, predicted_sim_stress, interpolatedStrain, wy1, wy2)
            fitnessScore = 1/candidateScore
            return fitnessScore
    elif CPLaw == "DB":
        def surrogateYieldBA(p, q, tausol):
            params = {
                'dipole': default_yield_value['dipole'],
                'islip': default_yield_value['islip'],
                'omega': default_yield_value['omega'],
                'p': p * (10 ** - param_range["p"]["round"]),
                'q': q * (10 ** - param_range["q"]["round"]), 
                'tausol': tausol * (10 ** - param_range["tausol"]["round"])
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            candidate_dict_round = round_params(params, param_range)
            solution = np.array(list(candidate_dict_round.values()))
            predicted_sim_stress = mlp.predict(solution.reshape(1, numberOfParams)).reshape(-1)
            candidateScore = fitness_yield(exp_target, predicted_sim_stress, interpolatedStrain, wy1, wy2)
            fitnessScore = 1/candidateScore
            return fitnessScore

    def ba_instance_run():
        # Initialize BA Optimizer
        
        ba_instance = bayes_opt.BayesianOptimization(f = surrogateYieldBA,
                                        pbounds = pbounds, verbose = 2,
                                        random_state = 4)
        
        # There are two ways of using BA: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # To use automatic way, comment out sequential way, from iterations = ... until after the loop
        # Sequential way  
        iterations = 200
        # Low kappa = 1 means more exploitation for UCB
        # High kappa = 10 means more exploration for UCB
        # Low xi = 0 means more exploitation for EI and POI
        # High xi = 0.1 means more exploration for EI and POI
        utility = bayes_opt.UtilityFunction(kind="ei", kappa=10, xi = 0.1)
        init_points = 200
        blockPrint()
        ba_instance.maximize(
            init_points = init_points, 
            n_iter = 0)
        for i in range(iterations):
            next_point = ba_instance.suggest(utility)
            target = surrogateYieldBA(**next_point)
            ba_instance.register(params=next_point, target=target)
            for param in next_point:
                original = next_point[param] * 10 ** - param_range[param]["round"]
                next_point[param] = original
            next_point = round_params(next_point, param_range)
            # print("#{} Result: {}; f(x) = {}.".format(i, next_point, target))
        enablePrint()
        '''
        # Automatic way
        init_points = 100
        iterations = 200
        #blockPrint()
        for i in range(1):
            ba_instance.maximize(
                init_points = init_points, 
                n_iter = iterations,    
                # What follows are GP regressor parameters
                acq="ucb", kappa=1, a=1)
        #enablePrint()
        ba_instance.set_gp_params(normalize_y=True)
        '''
        return ba_instance
    print("The experimental yield stress is: ", exp_target[0], "MPa")
    rangeSimYield = (exp_target[0]* (1 - yieldStressDev * 0.01), exp_target[0] * (1 + yieldStressDev * 0.01)) 
    print("The simulated yield stress should lie in the range of", rangeSimYield, "MPa")
    print("Maximum deviation:", exp_target[0] * yieldStressDev * 0.01, "MPa")
    print("#### Iteration", sim.fileNumber, "####")
    y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
    
    # If you want to find the best result from the initial random initial sims, you can set to true. It is likely that
    # one of the initial sims have yield stress close to the experimental yield stress so you can save time optimizing the yield stress
    bestResultFromInitialSimsLucky = False # You can change this
    
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
        ba_instance = ba_instance_run()
        partialResults = output_resultsPartialBA(ba_instance, param_range, default_yield_value, CPLaw)
        while tuple(partialResults['solution']) in sim.simulations.keys():
            print("The predicted solution is:")
            print(partialResults["solution_dict"])
            print("Parameters already probed. Algorithm needs to run again to obtain new parameters")
            ba_instance = ba_instance_run()
            partialResults = output_resultsPartialBA(ba_instance, param_range, default_yield_value, CPLaw)
        print_resultsPartialBA(partialResults)
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

def HardeningOptimizationBA(hardeningOptimizeInfo):
    # -------------------------------
    #      Initialize BA
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
    param_range_no_step = hardeningOptimizeInfo["param_range_no_step"] 
    exp_target = hardeningOptimizeInfo["exp_target"] 
    interpolatedStrain = hardeningOptimizeInfo["interpolatedStrain"] 
    sim = hardeningOptimizeInfo["sim"] 
    mlp = hardeningOptimizeInfo["mlp"] 
    partialResult = hardeningOptimizeInfo["partialResult"] 
    wh1 = weightsHardening["wh1"]
    wh2 = weightsHardening["wh2"]
    wh3 = weightsHardening["wh3"]
    wh4 = weightsHardening["wh4"]

    # -------------------------------
    #      Initialize BA
    # -------------------------------
    if CPLaw == "PH":
        pbounds = {
            "a": multiply(param_range_no_step['a'], 10 ** param_range["a"]["round"]), 
            "h0": multiply(param_range_no_step['h0'], 10 ** param_range["h0"]["round"]), 
            "taucs": multiply(param_range_no_step['taucs'], 10 ** param_range["taucs"]["round"])
        }
    elif CPLaw == "DB":
        pbounds = {
            "dipole": multiply(param_range_no_step['dipole'], 10 ** param_range["dipole"]["round"]), 
            "islip": multiply(param_range_no_step['islip'], 10 ** param_range["islip"]["round"]), 
            "omega": multiply(param_range_no_step['omega'], 10 ** param_range["omega"]["round"])
        }

    # Initialize surrogate function
    if CPLaw == "PH":
        def surrogateHardeningBA(a, h0, taucs):
            params = {
                'a': a * (10 ** - param_range["a"]["round"]),
                'h0': h0 * (10 ** - param_range["h0"]["round"]),
                'tau0': partialResult["tau0"],
                'taucs': taucs * (10 ** - param_range["taucs"]["round"])
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            candidate_dict_round = round_params(params, param_range)
            solution = np.array(list(candidate_dict_round.values()))
            predicted_sim_stress = mlp.predict(solution.reshape(1, numberOfParams)).reshape(-1)
            candidateScore = fitness_hardening(exp_target, predicted_sim_stress, interpolatedStrain, wh1, wh2, wh3, wh4)
            fitnessScore = 1/candidateScore
            return fitnessScore
    elif CPLaw == "DB":
        def surrogateHardeningBA(dipole, islip, omega):
            params = {
                'dipole': dipole * (10 ** - param_range["dipole"]["round"]),
                'islip': islip * (10 ** - param_range["islip"]["round"]),
                'omega': omega * (10 ** - param_range["omega"]["round"]),
                'p': partialResult["p"],
                'q': partialResult["q"],
                'tausol': partialResult["tausol"]
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            candidate_dict_round = round_params(params, param_range)
            solution = np.array(list(candidate_dict_round.values()))
            predicted_sim_stress = mlp.predict(solution.reshape(1, numberOfParams)).reshape(-1)
            candidateScore = fitness_hardening(exp_target, predicted_sim_stress, interpolatedStrain, wh1, wh2, wh3, wh4)
            fitnessScore = 1/candidateScore
            return fitnessScore


    
    # There are two ways of using BA: the sequential or automatic way. 
    # To use sequential way, comment out automatic way, from init_points = ... until after the loop
    # To use automatic way, comment out sequential way, from iterations = ... until after the loop
    def ba_instance_run():
        # Initialize BA Optimizer
        
        ba_instance = bayes_opt.BayesianOptimization(f = surrogateHardeningBA,
                                        pbounds = pbounds, verbose = 2,
                                        random_state = 4)
        
        # There are two ways of using BA: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # To use automatic way, comment out sequential way, from iterations = ... until after the loop
        # Sequential way  
        iterations = 200
        # Low kappa = 1 means more exploitation for UCB
        # High kappa = 10 means more exploration for UCB
        # Low xi = 0 means more exploitation for EI and POI
        # High xi = 0.1 means more exploration for EI and POI
        utility = bayes_opt.UtilityFunction(kind="ei", kappa=10, xi = 0.1)
        init_points = 200
        blockPrint()
        ba_instance.maximize(
            init_points = init_points, 
            n_iter = 0)
        for i in range(iterations):
            next_point = ba_instance.suggest(utility)
            target = surrogateHardeningBA(**next_point)
            ba_instance.register(params=next_point, target=target)
            for param in next_point:
                original = next_point[param] * 10 ** - param_range[param]["round"]
                next_point[param] = original
            next_point = round_params(next_point, param_range)
            # print("#{} Result: {}; f(x) = {}.".format(i, next_point, target))
        enablePrint()
        '''
        # Automatic way
        init_points = 100
        iterations = 200
        #blockPrint()
        for i in range(1):
            ba_instance.maximize(
                init_points = init_points, 
                n_iter = iterations,    
                # What follows are GP regressor parameters
                acq="ucb", kappa=1, a=1)
        #enablePrint()
        ba_instance.set_gp_params(normalize_y=True)
        '''
        return ba_instance
    print("The partial result: ")
    print(partialResult)
    y = np.array([interpolatedStressFunction(simStress, simStrain, interpolatedStrain) * convertUnit for (simStrain, simStress) in sim.simulations.values()])
    fullResult = list(sim.simulations.keys())[-1]
    print("The initial candidate full result: ")
    print(fullResult)
    # Iterative optimization.
    while not insideHardeningDev(exp_target, y[-1], hardeningDev):
        print("#### Iteration", sim.fileNumber + 1, "####")
        ba_instance = ba_instance_run()
        fullResults = output_resultsFullBA(ba_instance, param_range, partialResult, CPLaw)
        while tuple(fullResults['solution']) in sim.simulations.keys():
            print("The predicted solution is:")
            print(fullResults["solution_dict"])
            print("Parameters already probed. Algorithm needs to run again to obtain new parameters")
            ba_instance = ba_instance_run()
            fullResults = output_resultsFullBA(ba_instance, param_range, partialResult, CPLaw)
        print_resultsFullBA(fullResults)
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