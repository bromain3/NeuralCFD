import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.soot_transport.AUSM import AUSMFlux
from NeuralCFD.soot_transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.soot_transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation import FrenklachetWang_Leung_nucleation
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation_CPU import FrenklachetWang_Leung_nucleation_CPU
import NeuralCFD.solver.solver_soot_transport_soot_chemistry as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.read_field import read_field
import cantera as ct

import torch
import numpy as np
import random
#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change

def phi2Y(phi):
    MMCH4 = 16.04
    MMO2 = 31.999
    O2_N2 = 3.32

    mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
    Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
    return (Y)


def compute_free_flame(gas, refine_criterion=(3.0, 0.05, 0.1), width=0.03):
    flame = ct.FreeFlame(gas, width=width)

    flame.set_refine_criteria(ratio=refine_criterion[0], slope=refine_criterion[1], curve=refine_criterion[2])

    flame.solve(loglevel=0, refine_grid=True, auto=True)

    return (flame)

if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cuda:1'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    nclasses = 40

    th_file = r'../../input_data/gri30.yaml'

    frozen_flame_path = '../../simulation_results/soot_model_validation/refined_frozen_phi_2_20_flame.json'


    soot_d_min = 1.0e-9
    soot_d_max = 5 * 100.0e-9

    #####################################
    ########## object creation ##########
    #####################################
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    print()
    print(f'Thermo created with file: {th_file}')
    print()

    U, x = read_field(frozen_flame_path, thermo)

    grid = NeuralCFD.grid.grid.grid(float(x[-1] - x[0]), len(x), thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses, 'spacing': 'log'})

    print()
    if nclasses > 0:
        print(f'Grid created with sectional model enabled. Sectional model params are d_min={soot_d_min} and d_max={soot_d_max} with nclasses={nclasses}')
    else:
        print(f'Grid created with sectional model disabled.')
    print()

    Ns = torch.zeros((U.shape[1], nclasses), device=device, dtype=dtype).T
    U = torch.vstack((U, Ns))

    #####################################
    ######## boundary conditions ########
    #####################################
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U[:, 0]}


    ######################################
    ######### initial conditions #########
    ######################################
    U0 = U


    #######################################
    ######## solvers Configuration ########
    #######################################



    sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    sootTransport = rusanovFlux()

    solver = s.solver(grid, thermo, sootTransport, sootSolver, bc, U0, savingInterval=10
                      , savingPath='../../simulation_results/soot_model_validation')

    #solver.solve(0.00033, 100_000, 0.95, 'F_XU_comparison_stab_with_soot.json')


    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation', 'F_XU_comparison_stab_with_soot.json')
    if nclasses > 0:
        post_proc.animateField('T')
        #post_proc.plotSpatialSootEvolution2D(-1)
        #post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')