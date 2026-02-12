import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.soot_transport.AUSM import AUSMFlux
from NeuralCFD.soot_transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.soot_transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation import FrenklachetWang_Leung_nucleation
from NeuralCFD.soot_chemistry.oneWay_Leung_Lindset import oneWay_Leung_Lindset
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL_coarse
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.coagulation.Fuchs import Fuchs
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.SootModel import SootModel
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation_CPU import FrenklachetWang_Leung_nucleation_CPU
import NeuralCFD.solver.solver_soot_transport_soot_chemistry as s
import NeuralCFD.solver.solver_soot_transport_soot_chemistry_subcycle as ssub
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
from NeuralCFD.utilities_functions.read_field import read_field
import cantera as ct


import torch
import numpy as np
import random
#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change


if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cuda:1'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    ncells = 5000
    grid_length = 25.0e-3
    nclasses = 40

    th_file = r'../../input_data/gri30.yaml'
    field_path = '../../simulation_results/soot_model_validation/F_XU_comparison_cantera_soot_sootlib.json'


    ox_inflow_mdot = 139.2 * 1e-6
    fuel_inflow_mdot = 40.1 * 1e-6
    equivalence_ratio = 2.30

    burner_diameter = 60.0 * 1e-3
    inflow_T = 294.0
    inflow_P = 101325.0

    soot_d_min = 1.0e-9
    soot_d_max = 10_000.0e-9

    #############################################################
    ########## check if grid is smaller than particles ##########
    #############################################################
    print()
    print(f'Biggest particle diameter: {soot_d_max:.5e} m')
    print(f'cell size: {grid_length / ncells:.5e} m')
    print()

    #####################################
    ########## object creation ##########
    #####################################
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    print()
    print(f'Thermo created with file: {th_file}')
    print()
    U, x = read_field(field_path, thermo)

    grid = NeuralCFD.grid.grid.grid(float(x[-1] - x[0]), len(x), thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses, 'spacing': 'log'})

    print()
    if nclasses > 0:
        print(
            f'Grid created with sectional model enabled. Sectional model params are d_min={soot_d_min} and d_max={soot_d_max} with nclasses={nclasses}')
    else:
        print(f'Grid created with sectional model disabled.')
    print()



    #####################################
    ######## boundary conditions ########
    #####################################
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U[:, 0].flatten()}


    #######################################
    ######## solvers Configuration ########
    #######################################



    #sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    #sootSolver = oneWay_Leung_Lindset(grid, thermo)
    sootSolver = SootModel(nucLL, oxiLL, growthLL_coarse, FreeMolecular, grid, thermo)
    sootTransport = AUSMPlusUpFlux() #rusanovFlux()

    solver = ssub.solver(grid, thermo, sootTransport, sootSolver, bc, U, savingInterval=1, simdt=1e-5
                      , savingPath='../../simulation_results/soot_model_validation')

    solver.solve(10 * 0.0025, 10, 1.5, 'F_XU_comparison_cantera_soot_sootlib_new_growth.json')


    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation', 'F_XU_comparison_cantera_soot_sootlib_new_growth.json')
    if nclasses > 0:
        post_proc.animateField('T')
        post_proc.plotSpatialSootEvolution2D(-1)
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1, log_scale=True)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')