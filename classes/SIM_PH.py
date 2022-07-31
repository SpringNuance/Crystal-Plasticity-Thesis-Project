import os
import numpy as np
import random
import shutil
from .preprocessing import *


class SIM_PH:
    def __init__(
        self,
        info=None
    ):
        self.filename = 0
        self.filename2params = {}
        self.simulations = {}
        self.strain = None
        self.info=info
    
    def submit_array_jobs(self, start=None):
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
        shutil.copytree(f"./template_{material}/PH", path) # <== Path to DAMASK simulation setup folder.
        self.filename2params[path] = params
        self.edit_material_parameters(params, path)
    
    def edit_material_parameters(self, params, job_path):
        # Edit the material.config file.
        def a_edit(num):
            return f'a_slip                  {num}\n'

        def h0_edit(num):
            return f'h0_slipslip             {num}\n'

        def tau0_edit(num):
            return f'tau0_slip               {num}         # per family\n'

        def tausat_edit(num):
            return f'tausat_slip             {num}         # per family\n'

        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()

        lines[33] = a_edit(params[0])
        lines[34] = h0_edit(params[1])
        lines[31] = tau0_edit(params[2])
        lines[32] = tausat_edit(params[3])
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
            path = f'./simulations_{material}/PH{curveIndex}_{algorithm}/{str(self.filename)}'
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
        path = f'./simulations_{material}/PH{curveIndex}_{algorithm}/{str(self.filename)}'
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
            alpha = round(self.randrange_float(self.param_range['alpha']['low'], self.param_range['alpha']['high'], self.param_range['alpha']['step']), self.param_range['alpha']['round'])
            h0 = round(self.randrange_float(self.param_range['h0']['low'], self.param_range['h0']['high'], self.param_range['h0']['step']), self.param_range['alpha']['round'])
            tau = round(self.randrange_float(self.param_range['tau']['low'], self.param_range['tau']['high'], self.param_range['tau']['step']), self.param_range['alpha']['round'])
            taucs = round(self.randrange_float(self.param_range['taucs']['low'], self.param_range['taucs']['high'], self.param_range['taucs']['step']), self.param_range['alpha']['round'])
            points.append((alpha, h0, tau, taucs))
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