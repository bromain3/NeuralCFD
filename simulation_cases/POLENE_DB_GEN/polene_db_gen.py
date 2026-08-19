import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.POLENE_database_generator as db_gen
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
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

    def run_simulation_hotspot(phi, T, P, T0):

        t0 = T0
        print('phi =', phi, 'T =', T, 'P =', P, 'T0 =', T0)

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

        thermo = NeuralCFD.thermo.vectorized_thermo.thermo('../../input_data/gri30.yaml', device=device, dtype=dtype)
        grid = NeuralCFD.grid.grid.grid(15.0e-3 / 2, 1500, thermo, device=device, dtype=dtype)

        gas = ct.Solution(thermo.thermoFilePath)
        #bc = {'type': 'periodic'} #for base sim
        bc = {'type': 'symetric'}

        gas.Y = phi2Y(phi)

        p0 = P + 0.0 * grid.x
        u0 = 0.0 + 0.0 * grid.x
        T0 =  T0 + (T - T0) * torch.exp(-( grid.x / (18.0e-4) ) ** 2)  # hot spot in center

        Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
        ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
        for spe in gas.species_names:
            Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
        Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
        rho0 = p0 / Rspec0 / T0
        U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)

        solver = db_gen.solver(grid, thermo, bc, U0, savingPath='../../simulation_results/hotspot', reactorModel='cst_rho', printAdvance=True, savingInterval=1000000)
        solver.solve(3.2e-4, 3500, 1.0e-7, 0.95, f'sim_symetric_phi_{phi}_T_{T}_P_{P}_T0_{t0}.h5') # 3500




        #post_proc = pp.PostProcessing('../../simulation_results/hotspot', f'sim_symetric_phi_{phi}_T_{T}_X.json')
        #post_proc = pp.PostProcessing('../../simulation_results/hotspot', f'sim_symetric_phi_{phi}_T_{T}_P_{P}_T0_{t0}.h5')
        #post_proc = pp.PostProcessing('../../simulation_results/hotspot',f'sim_symetric_phi_1.0_T_2100_1500_cells_64bits_cantera_high_num_tol.h5')
        #post_proc.animateField('Y_NNH', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_HCO', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_C2H4', False, logScale=True, logThreshold=1e-20)
        #post_proc.animateField('Y_CH2(S)', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_CH', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_CH2', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_CH2OH', False, logScale=True, logThreshold=1e-28)
        #post_proc.animateField('Y_AR', True)
        #post_proc.animateField('Y_C2H2', logScale=True, logThreshold=1e-20)
        #post_proc.animateField('T')
        #post_proc.animateField('P')


        #post_proc.plotSnapchot('T', 0)
        #post_proc.animateField('P')
        #post_proc.plotSnapchot('P', 0)


    #Parallel(n_jobs=10, backend='loky')(delayed(run_simulation_hotspot)(phi) for phi in tqdm.tqdm(np.linspace(0.9, 1.1, 10)))
    #run_simulation_hotspot(1.05, 2050.0, 106391.25, 700.0)
    delta = 0.001
    phii = 1.0
    TT = 2000
    PP = 101325.0
    TT0 = 700

    done = set()

    try:
        with open("sim_done.txt", "r") as f:
            for line in f:
                done.add(line.strip())
    except FileNotFoundError:
        pass

    #run_simulation_hotspot(phii, TT, PP, TT0)
    with open('sim_done.txt', 'a') as f:
        for ph in np.linspace(phii * (1-delta), phii * (1+delta), 3):
            for t in np.linspace(TT * (1-delta), TT * (1+delta), 3):
                for p in np.linspace(PP * (1-delta), PP * (1+delta), 3):
                    for t0 in np.linspace(TT0 * (1-delta), TT0 * (1+delta), 3):
                        sim = (
                            f'phi = {ph}; T = {t}; '
                            f'P = {p}; T0 = {t0}'
                        )

                        if sim in done:
                            continue

                        run_simulation_hotspot(ph, t, p, t0)

                        f.write(sim + '\n')
                        f.flush()
                        done.add(sim)