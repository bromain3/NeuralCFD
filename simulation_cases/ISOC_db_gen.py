import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.POLENE_database_generator as db_gen
import NeuralCFD.solver.solver_transport_rusanov_cantera_chemestry as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct

import torch
import random
from joblib import Parallel, delayed
import tqdm
import sys
import numpy as np

#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change
if __name__ == "__main__":

    def run_simulation():

        def phi2Y(phi):
            MMCH4 = 16.04
            MMO2 = 31.999
            O2_N2 = 3.32

            mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
            Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
            return (Y)

        device = 'cpu'
        dtype = torch.float64

        torch.set_grad_enabled(False)

        thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml', device=device, dtype=dtype)
        grid = NeuralCFD.grid.grid.grid(10.0e-2, 200, thermo, device=device, dtype=dtype)

        gas = ct.Solution(thermo.thermoFilePath)
        bc = {'type': 'closed'}
        T_min = 1600.0
        T_max = 1800.0
        phi_min = 0.7
        phi_max = 1.5
        #300.0 + 300.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02 ** 2))  # hot spot in center

        T_left = T_min #random.uniform(T_min, T_max)
        T_right = T_max #random.uniform(T_left, T_max)
        phi = 1.0 #random.uniform(phi_min, phi_max)
        gas.Y = phi2Y(phi)
        #print(T_left, T_right, phi)

        p0 = 101325.0 + 0.0 * grid.x
        u0 = 0.0 + 0.0 * grid.x
        T0 =  T_left + (T_right - T_left) * (grid.x - min(grid.x)) / max(grid.x - min(grid.x))

        Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
        for spe in gas.species_names:
            Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
        Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
        rho0 = p0 / Rspec0 / T0
        U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

        #solver = s.solver(grid, thermo, bc, U0, savingInterval=10, savingPath='dtb_ISOC', reactorModel='cst_rho')
        solver = db_gen.solver(grid, thermo, bc, U0, savingPath='micro_db', reactorModel='cst_rho', printAdvance=True, savingInterval=1000000)
        solver.solve(1.8e-4, 1000000, 1e-7, 0.95, f'{T_left}_{T_right}_{phi}.json')

        post_proc = pp.PostProcessing('micro_db', '1600.0_1800.0_1.0_X.json')
        post_proc.animateField('T')
        #post_proc.plotVariable('T', 0)
        #post_proc.animateField('P')
        #post_proc.plotVariable('P', 0)


    def run_simulation_hotspot(phi):

        print('phi = ', phi)

        def phi2Y(phi):
            MMCH4 = 16.04
            MMO2 = 31.999
            O2_N2 = 3.32

            mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
            Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
            return (Y)

        device = 'cuda'
        dtype = torch.float64

        torch.set_grad_enabled(False)

        thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml', device=device, dtype=dtype)
        grid = NeuralCFD.grid.grid.grid(10.0e-3 / 2, 1000, thermo, device=device, dtype=dtype)

        gas = ct.Solution(thermo.thermoFilePath)
        #bc = {'type': 'periodic'} #for base sim
        bc = {'type': 'symetric'}

        gas.Y = phi2Y(phi)

        p0 = 101325.0 + 0.0 * grid.x
        u0 = 0.0 + 0.0 * grid.x
        T0 =  650.0 + (2000.0 - 650.0) * torch.exp(-( grid.x / (1.8 * 4 * 2.5e-4) ) ** 2)  # hot spot in center

        Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
        for spe in gas.species_names:
            Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
        Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
        rho0 = p0 / Rspec0 / T0
        U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

        solver = db_gen.solver(grid, thermo, bc, U0, savingPath='hotspot', reactorModel='cst_rho', printAdvance=True)
        solver.solve(2.9e-4, 3500, 1e-7, 0.95, f'sim_symetric_phi_{phi}.json')
        #solver.solve(5e-4, int(500 / 10), 10 * 1e-7, 0.95, f'sim_test.json')




        post_proc = pp.PostProcessing('hotspot', f'sim_symetric_phi_{phi}_X.json')
        #post_proc.animateField('Y_C2H2')
        #post_proc.plotVariable('T', 0)
        #post_proc.animateField('P')
        #post_proc.plotVariable('P', 0)

    #Parallel(n_jobs=10, backend='loky')(delayed(run_simulation_hotspot)(phi) for phi in tqdm.tqdm(np.linspace(0.9, 1.1, 10)))
    #for i in np.linspace(0.9, 1.1, 5):
    run_simulation_hotspot(1.0)