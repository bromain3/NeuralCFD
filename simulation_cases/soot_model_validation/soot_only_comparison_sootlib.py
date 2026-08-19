import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.soot_transport.AUSM import AUSMFlux
from NeuralCFD.soot_transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.soot_transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation import FrenklachetWang_Leung_nucleation
from NeuralCFD.soot_chemistry.oneWay_Leung_Lindset import oneWay_Leung_Lindset
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation_CPU import FrenklachetWang_Leung_nucleation_CPU
import NeuralCFD.solver.solver_soot_transport_soot_chemistry as s
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.growth.Lindset import Lindset as growthLIN
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.nucleation.Lindset import Lindset as nucLIN
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.coagulation.Fuchs_vect import Fuchs_vect
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.SootModel import SootModel
import NeuralCFD.solver.solver_soot_transport_soot_chemistry_subcycle as ssub
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
from NeuralCFD.utilities_functions.read_field import read_field
import cantera as ct
import pandas as pd


import torch
import numpy as np
import random
#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change


if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cuda:0'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    ncells = 5000
    grid_length = 25.0e-3
    nclasses = 40

    th_file = r'../../input_data/gri30.yaml'
    field_path = 'burner_profiles.inp'


    ox_inflow_mdot = 139.2 * 1e-6
    fuel_inflow_mdot = 40.1 * 1e-6
    equivalence_ratio = 2.30

    burner_diameter = 60.0 * 1e-3
    inflow_T = 294.0
    inflow_P = 101325.0

    # soot_d_min = 1.145853e-9 ; soot_d_max = 10_000.0e-9 # size used for LeungLindset model to ensure a nucleation of a particle at exactly 100 particles of carbon (if rho soot = 1800) as stated in the paper
    soot_d_min = 0.526715e-9  # size used for Lindset model to ensure a nucleation of a particle at exactly 10 particles of carbon (if rho soot = 1800) as stated in the paper (Naphtalene)
    soot_d_max = 10_000.0e-9  # size used for Lindset model to ensure a nucleation of a particle at exactly 10 particles of carbon (if rho soot = 1800) as stated in the paper (Naphtalene)

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

    with open(field_path) as f:
        header = f.readline().lstrip("#").split()

    data = pd.read_csv(
        field_path,
        sep='\s+',
        comment="#",
        names=header
    )
    ct_x_torch = torch.tensor(data['z(m)'], device=device, dtype=dtype)
    ct_u_torch = torch.tensor(data['v(m/s)'], device=device, dtype=dtype)
    ct_T_torch = torch.tensor(data['T(K)'], device=device, dtype=dtype)
    gas = ct.Solution(th_file)
    ct_Y_torch = {}
    for spe in gas.species_names:
        if f'y_{spe}' in data.columns:
            ct_Y_torch[spe] = torch.tensor(data[f'y_{spe}'], device=device, dtype=dtype)
        else:
            ct_Y_torch[spe] = torch.zeros_like(ct_T_torch)
    ct_Y_torch['AR'] = 1.0 - torch.sum(torch.stack([sp for sp in ct_Y_torch.values()]), dim=0)

    grid = NeuralCFD.grid.grid.grid(float(ct_x_torch[-1] - ct_x_torch[0]), ncells, thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses, 'spacing': 'log'})

    u0 = torch_interp(grid.x, ct_x_torch, ct_u_torch)
    T0 = torch_interp(grid.x, ct_x_torch, ct_T_torch)
    Y0 = {spe: torch_interp(grid.x, ct_x_torch, ct_Y_torch[spe]) for spe in gas.species_names}
    Y0 = torch.stack([Y0[spe] for spe in gas.species_names]).T
    p0 = inflow_P + 0.0 * grid.x
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Y0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Y0, ns0, thermo)



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
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U0[:, 0].flatten()}


    #######################################
    ######## solvers Configuration ########
    #######################################



    #sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    #sootSolver = oneWay_Leung_Lindset(grid, thermo)
    sootTransport = AUSMPlusUpFlux() #rusanovFlux()
    sootSolver = SootModel(nucLIN, oxiLL, growthLIN, FreeMolecular, grid, thermo)

    solver = ssub.solver(grid, thermo, sootTransport, sootSolver, bc, U0, savingInterval=160, simdt=1e-5, savingPath='../../simulation_results/soot_model_validation')

    #solver.solve(50 * 0.0025, 30_000, 1.5, 'F_XU_comparison_cantera_soot_sootlib_lin.json')


    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation', 'F_XU_comparison_cantera_soot_sootlib_lin.json')
    if nclasses > 0:
        #post_proc.animateField('T')
        #post_proc.plotSpatialSootEvolution2D(-1)
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1, log_scale=True)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')