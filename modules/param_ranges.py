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

##########################################################################
#                                                                        #
#                         MATERIAL RVE_1_40_D                            #
#                                                                        #
##########################################################################

# Parameter ranges for RVE_1_40_D
param_ranges_RVE_1_40_D = {

# Phenomenological parameters
    "PH": [
    # PH1 curve
    {
        'alpha':{'low': 1.1, 'high': 3, 'step': 0.1, 'round': 1},
        'h0':{'low': 100, 'high': 5000, 'step': 150, 'round': 0},
        'tau0':{'low': 20, 'high': 100, 'step': 1, 'round': 0},
        'taucs':{'low': 100, 'high': 1000, 'step': 10, 'round': 0} 
    },
    # PH2 curve
    {
        'alpha':{'low': 1.1, 'high': 3, 'step': 0.1, 'round': 1},
        'h0':{'low': 100, 'high': 1000, 'step': 50, 'round': 0},
        'tau0':{'low': 50, 'high': 100, 'step': 1, 'round': 0},
        'taucs':{'low': 100, 'high': 600, 'step': 1, 'round': 0} 
    },
    # PH3 curve
    {
        'alpha':{'low': 0.1, 'high': 0.9, 'step': 0.1, 'round': 1},
        'h0':{'low': 500, 'high': 1500, 'step': 50, 'round': 0},
        'tau0':{'low': 10, 'high': 50, 'step': 1, 'round': 0},
        'taucs':{'low': 20, 'high': 300, 'step': 1, 'round': 0}
    }],

# Dislocation-based parameters
    "DB" : [ 
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
}

# Default hardening parameters to optimize the yield stress parameters for RVE_1_40_D
default_yield_RVE_1_40_D = {

# Phenomenological parameters
    "PH": [
        {'alpha': 1.5,'h0': 800,'taucs': 200}, # PH1 curve
        {'alpha': 1.5,'h0': 800,'taucs': 200}, # PH2 curve
        {'alpha': 0.5,'h0': 800,'taucs': 200}  # PH3 curve
    ],

# Dislocation-based parameters
    "DB": [
        {'dipole': 5,'islip': 80, 'omega': 5}, # DB1 curve
        {'dipole': 5,'islip': 80, 'omega': 5}, # DB2 curve
        {'dipole': 5,'islip': 80, 'omega': 5}  # DB3 curve
    ],
}

############################################################################
#                                                                          #
#                         MATERIAL 512grains512                            #
#                                                                          #
############################################################################

# Parameter ranges for 512grains512
param_ranges_512grains512 = {

# Phenomenological parameters
    "PH": [
    # PH1 curve
    {
        'alpha':{'low': 1.2, 'high': 5, 'step': 0.2, 'round': 1},
        'h0':{'low': 500, 'high': 1000, 'step': 50, 'round': 0},
        'tau0':{'low': 100, 'high': 200, 'step': 1, 'round': 0},
        'taucs':{'low': 200, 'high': 400, 'step': 1, 'round': 0} 
    },
    # PH2 curve
    {
        'alpha':{'low': 3, 'high': 4, 'step': 0.1, 'round': 1},
        'h0':{'low': 1100, 'high': 1300, 'step': 50, 'round': 0},
        'tau0':{'low': 350, 'high': 380, 'step': 1, 'round': 0},
        'taucs':{'low': 1150, 'high': 1250, 'step': 1, 'round': 0} 
    },
    # PH3 curve
    {
        'alpha':{'low': 2.1, 'high': 3.1, 'step': 0.2, 'round': 1},
        'h0':{'low': 30000, 'high': 50000, 'step': 2000, 'round': 0},
        'tau0':{'low': 660, 'high': 700, 'step': 5, 'round': 0},
        'taucs':{'low': 500, 'high': 1000, 'step': 50, 'round': 0}
    }],

# Dislocation-based parameters
    "DB" : [ 
    # DB1 curve
    {
        'dipole':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'islip':{'low': 10, 'high': 70, 'step': 5, 'round': 0},
        'omega':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'p':{'low': 0.05, 'high': 1, 'step': 0.05, 'round': 2}, 
        'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
        'tausol':{'low': 0.3, 'high': 0.8, 'step': 0.01, 'round': 2}
    },
    # DB2 curve
    {
        'dipole':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'islip':{'low': 10, 'high': 70, 'step': 5, 'round': 0},
        'omega':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'p':{'low': 0.05, 'high': 1, 'step': 0.05, 'round': 2}, 
        'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
        'tausol':{'low': 0.3, 'high': 0.8, 'step': 0.01, 'round': 2}
    },
    # DB3 curve
    {
        'dipole':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'islip':{'low': 10, 'high': 70, 'step': 5, 'round': 0},
        'omega':{'low': 1, 'high': 15, 'step': 1, 'round': 0},
        'p':{'low': 0.05, 'high': 1, 'step': 0.05, 'round': 2}, 
        'q':{'low': 1, 'high': 2, 'step': 0.05, 'round': 2},
        'tausol':{'low': 0.3, 'high': 0.8, 'step': 0.01, 'round': 2}
    }]
}

# Default hardening parameters to optimize the yield stress parameters for 512grains512

default_yield_512grains512 = {

# Phenomenological parameters
    "PH": [
        {'alpha': 2, 'h0': 1000, 'taucs': 250}, # PH1 curve
        {'alpha': 2, 'h0': 1000, 'taucs': 250}, # PH2 curve
        {'alpha': 2, 'h0': 1000, 'taucs': 250} # PH3 curve
    ],

# Dislocation-based parameters
    "DB": [
        {'dipole': 5,'islip': 40, 'omega': 5}, # DB1 curve
        {'dipole': 5,'islip': 40, 'omega': 5}, # DB2 curve
        {'dipole': 5,'islip': 40, 'omega': 5}  # DB3 curve
    ],
}