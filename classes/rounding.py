from math import *

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
