# -------------------------------------------------------------------
#   Define parameter ranges, step size and rounding.
# -------------------------------------------------------------------

# The smaller the range and step size, the more refined the calibration. 
# However, if they are small, there are many more results to probe
# Balance of the range and step size is beneficial for parameter calibration
# The rounding is just for python code to generates random parameters correctly
# If the step size has 2 decimals, round = 2
# If the step size has 1 decimal, round = 1
# If the step size is integer, round = 0

# Note: the parameters should be ordered alphabetically for Bayesian algorithm
# If the initial_simulations.npy or simulations.npy file do not contain the data 
# of the parameters alphabetically, please use helpernpy.ipynb file to rearrange the 
# order of parameters so that the parameters are in correct order. You can also use helper.npy
# to see the contents of npy file

# Please change for the paramater ranges below according to your range and step size.
# if you need to use only one, just change one, such as PH1, and then go to optimize.py 
# and set CPLaw = "PH" and curveIndex = 1, and safely ignore other ranges
# Change for simulations, genetic and bayesian with the correct index
# In other words, you need to change Phenomenological algorithm, Ph1 curve simulations,
# PH1 curve genetic and PH1 curve Bayesian

# Phenomenological parameters
# For simulations
param_ranges_PH = [
# PH1 curve
{
    'alpha':{'low': 1.1, 'high': 3, 'step': 0.1, 'round': 1},
    'h0':{'low': 100, 'high': 5000, 'step': 150, 'round': 0},
    'tau':{'low': 20, 'high': 100, 'step': 1, 'round': 0},
    'taucs':{'low': 100, 'high': 1000, 'step': 10, 'round': 0} 
},
# PH2 curve
{
    'alpha':{'low': 1.1, 'high': 3, 'step': 0.1, 'round': 1},
    'h0':{'low': 100, 'high': 1000, 'step': 50, 'round': 0},
    'tau':{'low': 50, 'high': 100, 'step': 1, 'round': 0},
    'taucs':{'low': 100, 'high': 600, 'step': 1, 'round': 0} 
},
# PH3 curve
{
    'alpha':{'low': 0.1, 'high': 0.9, 'step': 0.1, 'round': 1},
    'h0':{'low': 500, 'high': 1500, 'step': 50, 'round': 0},
    'tau':{'low': 10, 'high': 50, 'step': 1, 'round': 0},
    'taucs':{'low': 20, 'high': 300, 'step': 1, 'round': 0}
}]

# For genetic algorithm
param_ranges_no_round_PH = [
# PH1 curve    
{
    'alpha':{'low': 1.1, 'high': 3, 'step': 0.1},
    'h0':{'low': 100, 'high': 5000, 'step': 150},
    'tau':{'low': 20, 'high': 100, 'step': 1},
    'taucs':{'low': 100, 'high': 1000, 'step': 10} 
},
# PH2 curve
{
    'alpha':{'low': 1.1, 'high': 3, 'step': 0.1},
    'h0':{'low': 100, 'high': 1000, 'step': 50},
    'tau':{'low': 50, 'high': 100, 'step': 1},
    'taucs':{'low': 100, 'high': 600, 'step': 1} 
},
# PH3 curve
{
    'alpha':{'low': 0.1, 'high': 0.9, 'step': 0.1},
    'h0':{'low': 500, 'high': 1500, 'step': 50},
    'tau':{'low': 10, 'high': 50, 'step': 1},
    'taucs':{'low': 20, 'high': 300, 'step': 1}
}
]

# For Bayesian Algorithm
param_ranges_no_step_PH = [
# PH1 curve
{
    'alpha': (1.1, 3),
    'h0': (100, 5000),
    'tau': (20, 100),
    'taucs': (100, 1000) 
},
# PH2 curve
{
    'alpha': (1.1, 3),
    'h0': (100, 1000),
    'tau': (50, 100),
    'taucs': (100, 600) 
},
# PH3 curve
{
    'alpha': (0.1, 0.9),
    'h0': (500, 1500),
    'tau': (10, 50),
    'taucs': (20, 300)
}
]

# Dislocation-based parameters
# For simulations
param_ranges_DB = [ 
# DB1 curve
{
    'dipole':{'low': 1, 'high': 25, 'step': 1, 'round': 0},
    'islip':{'low': 1, 'high': 50, 'step': 1, 'round': 0},
    'omega':{'low': 1, 'high': 50, 'step': 1, 'round': 0},
    'p':{'low': 0.05, 'high': 1, 'step': 0.05, 'round': 2}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
    'tausol':{'low': 0.01, 'high': 0.5, 'step': 0.01, 'round': 2}
},
# DB2 curve
{
    'dipole':{'low': 0.01, 'high': 1, 'step': 0.01, 'round': 2},
    'islip':{'low': 50, 'high': 100, 'step': 1, 'round': 0},
    'omega':{'low': 0.01, 'high': 1, 'step': 0.01, 'round': 2},
    'p':{'low': 0.01, 'high': 0.5, 'step': 0.01, 'round': 2}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
    'tausol':{'low': 1, 'high': 3, 'step': 0.05, 'round': 2}
},
# DB3 curve
{
    'dipole':{'low': 1, 'high': 25, 'step': 1, 'round': 0},
    'islip':{'low': 100, 'high': 200, 'step': 1, 'round': 0},
    'omega':{'low': 1, 'high': 50, 'step': 1, 'round': 0},
    'p':{'low': 0.05, 'high': 1, 'step': 0.05, 'round': 2}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
    'tausol':{'low': 1, 'high': 3, 'step': 0.05, 'round': 2}
}]

# For genetic algorithm
param_ranges_no_round_DB = [ 
# DB1 curve
{
    'dipole':{'low': 1, 'high': 25, 'step': 1},
    'islip':{'low': 1, 'high': 50, 'step': 1},
    'omega':{'low': 1, 'high': 50, 'step': 1},
    'p':{'low': 0.05, 'high': 1, 'step': 0.05}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05},
    'tausol':{'low': 0.01, 'high': 0.5, 'step': 0.01}
},
# DB2 curve
{
    'dipole':{'low': 0.01, 'high': 1, 'step': 0.01},
    'islip':{'low': 50, 'high': 100, 'step': 1},
    'omega':{'low': 0.01, 'high': 1, 'step': 0.01},
    'p':{'low': 0.01, 'high': 0.5, 'step': 0.01}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05},
    'tausol':{'low': 1, 'high': 3, 'step': 0.05}
},
# DB3 curve
{
    'dipole':{'low': 1, 'high': 25, 'step': 1},
    'islip':{'low': 100, 'high': 200, 'step': 1},
    'omega':{'low': 1, 'high': 50, 'step': 1},
    'p':{'low': 0.05, 'high': 1, 'step': 0.05}, 
    'q':{'low': 1, 'high': 2, 'step': 0.05},
    'tausol':{'low': 1, 'high': 3, 'step': 0.05}
}
]

# For Bayesian algorithm
param_ranges_no_step_DB = [ 
# DB1 curve
{
    'dipole': (1, 25),
    'islip': (1, 50),
    'omega': (1, 50),
    'p': (0.05, 1), 
    'q': (1, 2),
    'tausol': (0.01, 0.5)
},
# DB2 curve
{
    'dipole':(0.01, 1),
    'islip':(50, 100),
    'omega':(0.01, 1),
    'p':(0.01, 0.5), 
    'q':(1, 2),
    'tausol':(1, 3)
},
# DB3 curve
{
    'dipole': (1, 25),
    'islip': (100, 200),
    'omega': (1, 50),
    'p': (0.05, 1), 
    'q': (1, 2),
    'tausol': (1,3)
}]