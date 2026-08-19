import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_chemistry_subcycle as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.chemestry.neuralNetworkNewFormalism import neuralNetwork
import cantera as ct

import torch
import random
from joblib import Parallel, delayed
import tqdm
import sys
import numpy as np
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change
if __name__ == "__main__":
    def run_simulation_hotspot(phi, T):

        print('phi =', phi, 'T =', T)

        def phi2Y(phi):
            MMCH4 = 16.04
            MMO2 = 31.999
            O2_N2 = 3.32

            mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
            Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
            return (Y)

        device = 'cuda:0'
        dtype = torch.float64

        torch.set_grad_enabled(False)
        thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../../input_data/gri30.yaml', device=device, dtype=dtype, reducedSpeciesSet=["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "CH3", "CH4", "CO", "CO2", "CH2O", "CH3O", "CH3OH", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH", "N", "NH", "NH2", "NH3", "NO", "NO2", "N2O", "HNO", "HCN", "H2CN", "HCNO", "HOCN", "HNCO", "NCO", "N2", "AR", "C3H7", "C3H8", "CH2CHO", "CH3CHO", "bin"])
        #thermo = NeuralCFD.thermo.vectorized_thermo.thermo('../../input_data/gri30.yaml', device=device, dtype=dtype)
        grid = NeuralCFD.grid.grid.grid(15.0e-3 / 2, 1500, thermo, device=device, dtype=dtype)

        gas = ct.Solution(thermo.thermoFilePath)
        #bc = {'type': 'periodic'} #for base sim
        bc = {'type': 'symetric'}

        gas.Y = phi2Y(phi)

        p0 = 101325.0 + 0.0 * grid.x
        u0 = 0.0 + 0.0 * grid.x
        T0 =  700.0 + (T - 700.0) * torch.exp(-( grid.x / (18.0e-4) ) ** 2)  # hot spot in center

        Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
        ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
        Ys0[:, thermo.thermoProp.species.index('CH4')] = gas.Y[gas.species_names.index('CH4')] + 0.0 * grid.x
        Ys0[:, thermo.thermoProp.species.index('O2')] = gas.Y[gas.species_names.index('O2')] + 0.0 * grid.x
        Ys0[:, thermo.thermoProp.species.index('N2')] = gas.Y[gas.species_names.index('N2')] + 0.0 * grid.x
        Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
        rho0 = p0 / Rspec0 / T0
        U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)

        transportFlux = rusanovFlux()
        chem = neuralNetwork('export_PYTHON_autoreg_trained.pln', device=device, dtype=torch.float64, thermoSpecies=thermo.thermoProp.species)

        solver = s.solver(grid, thermo, transportFlux, chem, bc, U0, savingInterval=1, simdt=1.0e-7, savingPath='../../simulation_results/Network1D', reactorModel='cst_rho', initialGridTemperature=T0)
        #solver.solve(3.2e-4, 849, 0.95, f'new_net.json') # 3500




        post_proc = pp.PostProcessing('../../simulation_results/Network1D', f'new_net.json')
        post_proc.animateField('P', logScale=False)
        #post_proc.animateField('Y_N2')
        #post_proc.animateField('Y_C2H2')
        post_proc.plotSnapchot('T', 0)
        post_proc.plotSnapchot('T', -1)
        #post_proc.animateField('P')
        #post_proc.plotSnapchot('P', 0)

    #Parallel(n_jobs=10, backend='loky')(delayed(run_simulation_hotspot)(phi) for phi in tqdm.tqdm(np.linspace(0.9, 1.1, 10)))
    #for i in np.linspace(0.9, 1.1, 5):
    #run_simulation_hotspot(0.95)
    run_simulation_hotspot(1.0, 2000)