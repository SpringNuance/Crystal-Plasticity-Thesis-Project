import random
import numpy as np

param_range = {
    'tau':{'low': 20, 'high': 100, 'step': 1},
    'taucs':{'low': 100, 'high': 1000, 'step': 5}, 
    'h0':{'low': 100, 'high': 5000, 'step': 100},
    'alpha':{'low': 1, 'high': 3, 'step': 0.1}
}

def randrange_float(start, stop, step):
    spaces = int((stop - start) / step)
    return random.randint(0, spaces) * step + start

points = []
np.random.seed(20)
for _ in range(30):
    tau = randrange_float(param_range['tau']['low'], param_range['tau']['high'], param_range['tau']['step'])
    taucs = randrange_float(param_range['taucs']['low'], param_range['taucs']['high'], param_range['taucs']['step'])
    h0 = randrange_float(param_range['h0']['low'], param_range['h0']['high'], param_range['h0']['step'])
    alpha = round(randrange_float(param_range['alpha']['low'], param_range['alpha']['high'], param_range['alpha']['step']), 1)
    points.append((tau, taucs, h0,alpha))
print(points)

# python randomstep.py