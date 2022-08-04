from modules.helper import *

def output_resultsBA(ba_instance):
    # Returning the details of the best solution in a dictionary.
    solution = round_params(ba_instance.max["params"])
    solution_fitness = ba_instance.max["target"]
    fitness = 1/solution_fitness
    values = (solution, solution_fitness, fitness)
    keys = ("solution", "solution_fitness", "fitness")
    output = dict(zip(keys, values))
    return output

def print_resultsBA(results):
    print(f"Parameters of the best solution : {results['solution']}")
    print(f"Fitness value of the best solution = {results['solution_fitness']}")
    print(f"Fitness given by the MLP estimate: {results['fitness']}")


'''
elif algorithm == "BA":
    # -------------------------------
    #      Initialize BA
    # -------------------------------

    # Initialize surrogate function
    if CPLaw == "PH":
        def surrogate(alpha, h0, tau0, taucs):
            params = {
                'alpha': alpha,
                'h0': h0,
                'tau0': tau0,
                'taucs': taucs
            }
            # Rounding is required because BA only deals with continuous values.
            # Rounding help BA probe at discrete parameters with correct step size
            alpha, h0, tau0, taucs = round_params(params, param_range)
            solution = np.array([alpha, h0, tau0, taucs])
            predicted_sim_stress = mlp.predict(solution.reshape((1, numberOfParams))).reshape(-1)
            fitnessScore = fitness(exp_target, predicted_sim_stress, interpolatedStrain, w1, w2, w3, w4)
            fitnessScore = 1/fitnessScore
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
            fitnessScore = fitness(exp_target, predicted_sim_stress, interpolatedStrain, w1, w2, w3, w4)
            fitnessScore = 1/fitnessScore
            return fitnessScore

    # Initialize BA Optimizer
    ba_instance = bayes_opt.BayesianOptimization(f = surrogate,
                                    pbounds = param_range_no_step, verbose = 2,
                                    random_state = 4)
    
    # There are two ways of using BA: the sequential or automatic way. 
    # To use sequential way, comment out automatic way, from init_points = ... until after the loop
    # TO use automatic way, comment out sequential way, from iterations = ... until after the loop
    # Sequential way  
    def ba_instance_run():
        iterations = 100
        utility = bayes_opt.UtilityFunction(kind="ucb", kappa=2.5, xi = 1)
        for _ in range(iterations):
            next_point = ba_instance.suggest(utility)
            target = surrogate(**next_point)
            ba_instance.register(params=next_point, target=target)

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

    # -------------------------------
    #      End of BA
    # -------------------------------
'''

def YieldStressOptimizationBA():
    return 0
def HardeningOptimizationBA():
    return 0