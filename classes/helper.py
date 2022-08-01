from math import *
import copy

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