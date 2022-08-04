import os
import numpy as np
import random
import shutil
from .preprocessing import *


class SIM:
    def __init__(
        self,
        info=None
    ):
        self.fileNumber = 0
        self.fileNumber2params = {}
        self.simulations = {}
        self.strain = None
        self.info=info
    
    def submit_array_jobs(self, start=None):
        """
        Run the simulation and postprocessing.
        Array jobs will submit multiple simulations from starting number up until fileNumber.
        code = 0 if success
        code = 1 if errors occurred
        """
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        projectPath = self.info['projectPath']
        mode = self.info["mode"]
        fullpath = f"{projectPath}/simulations_{material}/{CPLaw}{curveIndex}_{algorithm}"
        if start:
            code = os.system(f'sh array_runsim_{mode}.sh {self.fileNumber} {start} {fullpath} {material}')
        else:
            code = os.system(f'sh array_runsim_{mode}.sh {self.fileNumber} {1} {fullpath} {material}')
        return code
    
    def submit_array_jobs_post(self, start=None):
        """
        Run the simulation and postprocessing.
        Array jobs will submit multiple simulations from starting number up until fileNumber.
        code = 0 if success
        code = 1 if errors occurred
        """
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        projectPath = self.info['projectPath']
        mode = self.info["mode"]
        fullpath = f"{projectPath}/simulations_{material}/{CPLaw}{curveIndex}_{algorithm}"
        if start:
            code = os.system(f'sh array_runsimpost_{mode}.sh {self.fileNumber} {start} {fullpath} {material}')
        else:
            code = os.system(f'sh array_runsimpost_{mode}.sh {self.fileNumber} {1} {fullpath} {material}')
        return code
    
    def make_new_job(self, params, path):
        material = self.info["material"]
        CPLaw = self.info['CPLaw']
        shutil.copytree(f"./template_{material}/{CPLaw}", path) # <== Path to DAMASK simulation setup folder.
        self.fileNumber2params[path] = params
        if self.info["CPLaw"] == "PH":
            if self.info["material"] == "RVE_1_40_D":
                self.edit_material_parameters_PH_RVE_1_40_D(params, path)
            elif self.info["material"] == "512grains512":
                self.edit_material_parameters_PH_512grains512(params, path)
        if self.info["CPLaw"] == "DB":
            if self.info["material"] == "RVE_1_40_D":
                self.edit_material_parameters_DB_RVE_1_40_D(params, path)
            elif self.info["material"] == "512grains512":
                self.edit_material_parameters_DB_512grains512(params, path)
    
    def edit_material_parameters_PH_RVE_1_40_D(self, params, job_path):
        # Edit the material.config file.
        def alpha_edit(num):
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
        index = self.info["editLinesPH"]
        lines[index[0]] = alpha_edit(params[0])
        lines[index[1]] = h0_edit(params[1])
        lines[index[2]] = tau0_edit(params[2])
        lines[index[3]] = tausat_edit(params[3])
        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)
    
    def edit_material_parameters_PH_512grains512(self, params, job_path):
        # Edit the material.config file.
        def alpha_edit(num):
            return f'a_slip                  {num}\n'

        def h0_edit(num):
            return f'h0_slipslip             {num}\n'

        def tau0_edit(num):
            return f'tau0_slip               {num} {num}        # per family\n'

        def tausat_edit(num):
            return f'tausat_slip             {num} {num}       # per family\n'

        path = f'{job_path}/material.config'
        with open(path) as f:
            lines = f.readlines()
        index = self.info["editLinesPH"]
        lines[index[0]] = alpha_edit(params[0])
        lines[index[1]] = h0_edit(params[1])
        lines[index[2]] = tau0_edit(params[2])
        lines[index[3]] = tausat_edit(params[3])

        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)

    def edit_material_parameters_DB_RVE_1_40_D(self, params, job_path):
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
        index = self.info["editLinesDB"]
        lines[index[0]] = dipole_edit(params[0])
        lines[index[1]] = islip_edit(params[1])
        lines[index[2]] = omega_edit(params[2])
        lines[index[3]] = p_edit(params[3])
        lines[index[4]] = q_edit(params[4])
        lines[index[5]] = tausat_edit(params[5])

        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)
    
    def edit_material_parameters_DB_512grains512(self, params, job_path):
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
        index = self.info["editLinesDB"]
        lines[index[0]] = dipole_edit(params[0])
        lines[index[1]] = islip_edit(params[1])
        lines[index[2]] = omega_edit(params[2])
        lines[index[3]] = p_edit(params[3])
        lines[index[4]] = q_edit(params[4])
        lines[index[5]] = tausat_edit(params[5])

        with open(f'{job_path}/material.config', 'w') as f:
            f.writelines(lines)

    def run_initial_simulations_auto(self):
        """
        Runs N simulations according to get_grid().
        Used when initializing a response surface.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        CPLaw = self.info['CPLaw']
        n_params = self.get_grid()
        simPath = f'./simulations_{material}/{CPLaw}{curveIndex}_{algorithm}'
        shutil.rmtree(simPath)
        os.mkdir(simPath)
        for params in n_params:
            self.fileNumber += 1
            fileNumber = str(self.fileNumber)
            path = f'{simPath}/{fileNumber}'
            self.make_new_job(params, path)
        self.submit_array_jobs()
        self.strain = self.save_outputs()

    def run_initial_simulations_manual(self, tupleParams):
        """
        Runs N simulations according to get_grid().
        Used when initializing a response surface.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        CPLaw = self.info['CPLaw']
        n_params = tupleParams[0 : self.info['initialSims']]
        np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_params.npy', n_params)
        simPath = f'./simulations_{material}/{CPLaw}{curveIndex}_{algorithm}'
        shutil.rmtree(simPath)
        os.mkdir(simPath)
        for params in n_params:
            self.fileNumber += 1
            fileNumber = str(self.fileNumber)
            path = f'{simPath}/{fileNumber}'
            self.make_new_job(params, path)
        self.submit_array_jobs()
        self.strain = self.save_outputs()
    
    def run_initial_simulations_post(self, tupleParams):
        """
        Runs N simulations according to get_grid().
        Used when initializing a response surface.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        CPLaw = self.info['CPLaw']
        n_params = tupleParams[0 : self.info['initialSims']]
        simPath = f'./simulations_{material}/{CPLaw}{curveIndex}_{algorithm}'
        n_params = np.load(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_params.npy', allow_pickle=True)
        n_params = n_params.tolist()
        for params in n_params:
            self.fileNumber += 1
            fileNumber = str(self.fileNumber)
            path = f'{simPath}/{fileNumber}'
            self.fileNumber2params[path] = params
        self.submit_array_jobs_post()
        self.strain = self.save_outputs()

    def run_single_test(self, params):
        """
        Runs a single simulation with 'params'.
        Used during optimization process.
        """
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        CPLaw = self.info['CPLaw']
        self.fileNumber += 1
        fileNumber = str(self.fileNumber)
        simPath = f'./simulations_{material}/{CPLaw}{curveIndex}_{algorithm}/{fileNumber}'
        if os.path.exists(simPath) and os.path.isdir(simPath):
            shutil.rmtree(simPath)
        self.make_new_job(params, simPath)
        self.submit_array_jobs(start=self.fileNumber)
        self.save_single_output(simPath, params)
    
    def randrange_float(self, start, stop, step):
        spaces = int((stop - start) / step)
        return random.randint(0, spaces) * step + start

    def get_grid(self):
        points = []
        np.random.seed(20)
        for _ in range(self.info['initialSims']):    
            candidateParam = []
            for parameter in self.info['param_range']:
                candidateParam.append(round(self.randrange_float(self.info['param_range'][parameter]['low'], self.info['param_range'][parameter]['high'], self.info['param_range'][parameter]['step']), self.info['param_range'][parameter]['round']))
            points.append(tuple(candidateParam))
        material = self.info["material"]
        curveIndex = self.info['curveIndex']
        algorithm = self.info['algorithm']
        CPLaw = self.info['CPLaw']
        np.save(f'results_{material}/{CPLaw}{curveIndex}_{algorithm}/initial_params.npy', points)
        return points
    
    def save_outputs(self):
        true_strains = []
        for (path, params) in self.fileNumber2params.items():
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