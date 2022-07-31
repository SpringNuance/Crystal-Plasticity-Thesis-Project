import os
import numpy as np
import random
import shutil
from .preprocessing import *


class SIM_DB:
    def __init__(
        self,
        info=None
    ):
        self.filename = 0
        self.filename2params = {}
        self.simulations = {}
        self.strain = None
        self.info=info
    
    def submit_array_jobs(self, info, start=None):
        """
        Run the simulation and postprocessing.
        Array jobs will submit multiple simulations up until filename:int.
        code = 0 if success
        code = 1 if errors occurred
        """
        if start:
            code = os.system(f'sh array_runsim.sh {self.filename} {start}')
        else:
            code = os.system(f'sh array_runsim.sh {self.filename} {1}')
        return code
    
    def make_new_job(self, params, path):
        material = self.info["material"]
        shutil.copytree(f"./template_{material}/DB", path) # <== Path to DAMASK simulation setup folder.
        self.filename2params[path] = params
        self.edit_material_parameters(params, path)
    
    def edit_material_parameters(self, params, job_path):
        # Edit the material.config file.
        def dipole_edit(num):
            return f'Cedgedipmindistance {num}             # Adj. parameter controlling the minimum dipole distance [in b]\n'

        def islip_edit(num):
            return f'CLambdaSlip         {num}           # Adj. parameter controlling dislocation mean free path\n'
        
        def omega_edit(num):
            return f'Catomicvolume       {num}             # Adj. parameter controlling the atomic volume [in b^3]\n'

        def p_edit(num):
            return f'p_slip              {num}            # p-exponent in glide velocity\n'

        def q_edit(num):
            return f'q_slip              {num}             # q-exponent in glide velocity\n'

        def tausat_edit(num):
            return f'SolidSolutionStrength {num}e8         # Strength due to elements in solid solution\n'

        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()
        lines[66] = dipole_edit(params[0])
        lines[62] = islip_edit(params[1])
        lines[65] = omega_edit(params[2])
        lines[58] = p_edit(params[3])
        lines[59] = q_edit(params[4])
        lines[49] = tausat_edit(params[5])

        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)
    
    def run_initial_simulations(self):
        """
        Runs N simulations according to get_grid().
        Used when initializing a response surface.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        n_params = self.get_grid()
        for params in n_params:
            self.filename += 1
            path = f'./simulations_{material}/DB{curveIndex}_{algorithm}/{str(self.filename)}'
            self.make_new_job(params, path)
        self.submit_array_jobs()
        self.strain = self.save_outputs()

    def run_single_test(self, params):
        """
        Runs a single simulation with 'params'.
        Used during optimization process.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        self.filename += 1
        path = f'./simulations_{material}/DB{curveIndex}_{algorithm}/{str(self.filename)}'
        self.make_new_job(params, path)
        self.submit_array_jobs(start=self.filename)
        self.save_single_output(path, params)
    
    def randrange_float(self, start, stop, step):
        spaces = int((stop - start) / step)
        return random.randint(0, spaces) * step + start

    def get_grid(self):
        points = []
        np.random.seed(20)
        for _ in range(self.initialSims):
            dipole = round(self.randrange_float(self.param_range['dipole']['low'], self.param_range['dipole']['high'], self.param_range['dipole']['step']), 0)
            islip = round(self.randrange_float(self.param_range['islip']['low'], self.param_range['islip']['high'], self.param_range['islip']['step']), 0)
            omega = round(self.randrange_float(self.param_range['omega']['low'], self.param_range['omega']['high'], self.param_range['omega']['step']), 0)
            p = round(self.randrange_float(self.param_range['p']['low'], self.param_range['p']['high'], self.param_range['p']['step']), 2)
            q = round(self.randrange_float(self.param_range['q']['low'], self.param_range['q']['high'], self.param_range['q']['step']), 2)
            tausol = round(self.randrange_float(self.param_range['tausol']['low'], self.param_range['tausol']['high'], self.param_range['tausol']['step']), 2)            
            points.append((dipole, islip, omega, p, q, tausol))
        return points
    
    def save_outputs(self):
        true_strains = []
        for (path, params) in self.filename2params.items():
            path2txt = f'{path}/postProc/'
            files = [f for f in os.listdir(path2txt) if os.path.isfile(os.path.join(path2txt, f))]
            print("path2txt is:", path2txt)
            print("files is: ")
            print(files)
            processed = preprocess(f'{path2txt}/{files[0]}')
            true_strains.append(processed[0])
            self.simulations[params] = processed
        return np.array(true_strains).mean(axis=0) # Outputs the strain values for the simulations

    def save_single_output(self, path, params):
        path2txt = f'{path}/postProc/'
        files = os.listdir(path2txt)
        print("path2txt is:", path2txt)
        print("files is: ")
        print(files)
        processed = preprocess(f'{path2txt}/{files[0]}')
        self.simulations[params] = processed