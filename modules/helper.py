from math import *
import copy
import numpy as np
from scipy.interpolate import interp1d
from modules.helper import *

def round_to_step(low, step, value, roundValue):
    upperBound = floor((value - low)/step) * step + low
    lowerBound = ceil((value - low)/step) *step + low
    upperDif = upperBound - value
    lowerDif = value - lowerBound
    if upperDif >= lowerDif:
        return round(upperBound, roundValue)
    else: 
        return round(lowerBound, roundValue)

# params and param_range are both dictionaries
def round_params(params, param_range):
    for parameter in params:
        params[parameter] = round_to_step(param_range[parameter]['low'], param_range[parameter]['step'], params[parameter], param_range[parameter]['round'])
    return params

# Requiring that the interpolatedStrain must lie inside the range of strain
def interpolatedStressFunction(stress, strain, interpolatedStrain):
    # interpolated function fits the stress-strain curve data 
    interpolatedFunction = interp1d(strain, stress)
    # Calculate the stress values at the interpolated strain points
    interpolatedStress = interpolatedFunction(interpolatedStrain)
    return interpolatedStress 

def param_range_no_round_func(param_range):
    param_range_no_round = {}
    temporary_param_range = copy.deepcopy(param_range)
    for key in param_range:
        temporary_param_range[key].pop('round')
        param_range_no_round[key] = temporary_param_range[key]
    return param_range_no_round

def param_range_no_step_func(param_range_no_round):
    param_range_no_step = {}
    temporary_param_range = copy.deepcopy(param_range_no_round)
    for key in param_range_no_round:
        temporary_param_range[key].pop('step')
        param_range_no_step[key] = (temporary_param_range[key]['low'], temporary_param_range[key]['high'])
    return param_range_no_step

def rearrangePH(params):
    newParams = {}
    newParams['alpha'] = params['alpha']
    newParams['h0'] = params['h0']
    newParams['tau0'] = params['tau0']
    newParams['taucs'] = params['taucs']
    return newParams
    
def rearrangeDB(params):
    newParams = {}
    newParams['dipole'] = params['dipole']
    newParams['islip'] = params['islip']
    newParams['omega'] = params['omega']
    newParams['p'] = params['p']
    newParams['q'] = params['q']
    newParams['tausol'] = params['tausol']
    return newParams

def tupleOrListToDict(params, CPLaw):
    newParams = {}
    if CPLaw == "PH":
        newParams['alpha'] = params[0]
        newParams['h0'] = params[1]
        newParams['tau0'] = params[2]
        newParams['taucs'] = params[3]
    if CPLaw == "DB":
        newParams['dipole'] = params[0]
        newParams['islip'] = params[1]
        newParams['omega'] = params[2]
        newParams['p'] = params[3]
        newParams['q'] = params[4]
        newParams['tausol'] = params[5]
    return newParams

def defaultParams(partialResult, CPLaw, default_yield_value):
    if CPLaw == "PH":
        solution_dict = {
            'alpha': default_yield_value['alpha'],
            'h0': default_yield_value['h0'],
            'tau0': partialResult['tau0'],
            'taucs': default_yield_value['taucs']
        }
    elif CPLaw == "DB":
        solution_dict = {
            'dipole': default_yield_value['dipole'],
            'islip': default_yield_value['islip'],
            'omega': default_yield_value['omega'],
            'p': partialResult["p"],
            'q': partialResult["q"], 
            'tausol': partialResult["tausol"]
        }
    return solution_dict

def getIndexBeforeStrainLevel(strain, level):
    for i in range(len(strain)):
        if strain[i] > level:
            return i - 1

def getIndexBeforeStrainLevelEqual(strain, level):
    for i in range(len(strain)):
        if strain[i] >= level:
            return i

def calculateInterpolatingStrains(mainStrain, limitingStrain, yieldStressStrainLevel, dropUpperEnd):
    x_max = limitingStrain.max() 
    indexUpper = getIndexBeforeStrainLevelEqual(mainStrain, x_max)
    indexLower = getIndexBeforeStrainLevel(mainStrain, yieldStressStrainLevel) 
    mainStrain = mainStrain[:indexUpper]
    mainStrain = mainStrain[indexLower:]
    # If the error: ValueError: A value in x_new is above the interpolation range occurs,
    # it is due to the the strain value of some simulated curves is higher than the last stress value
    # of the interpolated strain so it lies outside the range. You can increase the dropUpperEnd number to reduce the
    # range of the simulated curves so their stress can be interpolated
    # interpolatedStrain will be the interpolating strain for all curves (experimental, initial simulation and iterated simulation)
    interpolatedStrain = mainStrain
    if dropUpperEnd != 0:
        interpolatedStrain = mainStrain[:-dropUpperEnd]
    # Strain level is added to the interpolating strains
    interpolatedStrain = np.insert(interpolatedStrain, 1, yieldStressStrainLevel)
    return interpolatedStrain