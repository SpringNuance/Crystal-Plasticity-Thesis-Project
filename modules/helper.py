from math import *
import copy
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