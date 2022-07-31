import numpy as np

pathInitial = "CrystalPlasticityProject\results_RVE_1_40_D\PH2_GA\initial_simulations.npy"
pathSimulation = "CrystalPlasticityProject\results_RVE_1_40_D\PH2_GA\simulations.npy"
initial_data = np.load('initial_simulations.npy', allow_pickle=True)

initial_data = initial_data.tolist()
oldkeys = list(initial_data.keys())
# Type in i[number] to rearrange your parmeters
newkeys = list(map(lambda i: (i[3], i[2], i[0], i[1]), initial_data.keys()))
print(type(initial_data))
for i in range(len(newkeys)):
    initial_data[newkeys[i]] = initial_data.pop(oldkeys[i])
        
np.save('initial_simulations2.npy', initial_data)
initial_data2 = np.load('initial_simulations2.npy', allow_pickle=True)

initial_data = np.load('simulations.npy', allow_pickle=True)
print(initial_data)

initial_data = initial_data.tolist()
oldkeys = list(initial_data.keys())
newkeys = list(map(lambda i: (i[3], i[2], i[0], i[1]), initial_data.keys()))
print(oldkeys)
print(newkeys)
print(type(initial_data))
for i in range(len(newkeys)):
    initial_data[newkeys[i]] = initial_data.pop(oldkeys[i])
        
np.save('simulations2.npy', initial_data)
initial_data2 = np.load('simulations2.npy', allow_pickle=True)
# print(initial_data2)

# python rearrange.py