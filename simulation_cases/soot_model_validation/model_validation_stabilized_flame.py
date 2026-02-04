import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation import FrenklachetWang_Leung_nucleation
from NeuralCFD.soot_chemistry.FrenklachetWang_Leung_nucleation_CPU import FrenklachetWang_Leung_nucleation_CPU
from NeuralCFD.chemestry.canteraChemMultiproc import canteraChemMultiproc
import NeuralCFD.solver.solver_transport_chemistry_subcycle as s
import NeuralCFD.solver.solver_transport_chemistry_soot_subcycle as stcss
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct
# CRITICAL: Set thread limits BEFORE importing numpy/scipy
# This prevents OpenBLAS/MKL from spawning too many threads per process
#import os
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['NUMEXPR_NUM_THREADS'] = '1'
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
    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    ncells = 2000
    grid_length = 20.0e-3 / 2
    nclasses = 0

    th_file = r'../../input_data/gri30.yaml'

    inflow_fuel_equivalence_ratio = 2.20
    inflow_T = 294.0
    inflow_P = 101325.0

    soot_d_min = 1.0e-9
    soot_d_max = 100.0e-9

    #####################################
    ########## object creation ##########
    #####################################
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    print()
    print(f'Thermo created with file: {th_file}')
    print()

    grid = NeuralCFD.grid.grid.grid(grid_length, ncells, thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses, 'spacing': 'log'})

    print()
    if nclasses > 0:
        print(f'Grid created with sectional model enabled. Sectional model params are d_min={soot_d_min} and d_max={soot_d_max} with nclasses={nclasses}')
    else:
        print(f'Grid created with sectional model disabled.')
    print()

    ######################################
    ###### flamme speed computation ######
    ######################################

    print()
    print("Computing 1D flame with cantera for having flame speed")
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = inflow_T, inflow_P
    gas.set_equivalence_ratio(inflow_fuel_equivalence_ratio, fuel="CH4", oxidizer="O2")

    f = compute_free_flame(gas, width=grid_length)
    inflow_speed = f.velocity[0]
    print()


    #####################################
    ######## boundary conditions ########
    #####################################

    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = inflow_T, inflow_P
    gas.set_equivalence_ratio(inflow_fuel_equivalence_ratio, fuel="CH4", oxidizer="O2")

    P_inflow = inflow_P
    T_inflow = inflow_T
    u_inflow = inflow_speed

    T_in = torch.tensor([T_inflow], dtype=dtype, device=device)
    p_in = torch.tensor([P_inflow], dtype=dtype, device=device)
    u_in = torch.tensor([u_inflow], dtype=dtype, device=device)

    Ys_in = torch.zeros((1, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys_in[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)]
    Ns_in = torch.zeros((1, grid.nclasses), device=device, dtype=dtype)

    Rspec_in = thermo.thermoProp.R_spec_from_Y(Ys_in)
    rho_in = p_in / Rspec_in / T_in
    U_in = pc.primitive_to_conserved(rho_in, u_in, p_in, T_in, Ys_in, Ns_in, thermo).flatten()
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U_in}

    print()
    methane_masse_flow = U_in[3+ thermo.thermoProp.species.index('CH4')] * u_inflow
    oxygen_masse_flow = U_in[3 + thermo.thermoProp.species.index('O2')] * u_inflow
    print(f'Boundary conditions set with P={P_inflow}, T={T_inflow}, flame speed={u_inflow}, methane flow per unit area = {methane_masse_flow}, oxygen flow per unit area = {oxygen_masse_flow}')
    print(f'experiment baseline P=1 atm, T=???, flame speed=???, methane flow per unit area ~ {41.0e-6 / (torch.pi * (60e-3 / 2)**2)}, oxygen flow per unit area ~ {138.0e-6 / (torch.pi * (60e-3 / 2)**2)}')
    print()

    ######################################
    ######### initial conditions #########
    ######################################
    p0 = inflow_P + 0.0 * grid.x
    u0 = inflow_speed + 0.0 * grid.x
    T0 = 2500 + 0.0 * grid.x

    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)]


    p0 = inflow_P + 0.0 * grid.x
    u0 = inflow_speed + 0.0 * grid.x
    T0 = inflow_T + 0.0 * grid.x

    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)]

    gas.equilibrate('HP')
    T0[int(ncells / 2):] = gas.T
    for spe in gas.species_names:
        Ys0[int(ncells / 2):, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)]
    #for i in range(nclasses):
    #    ns0[int(ncells / 2):, i] = 1e14 * np.exp(1 - i / nclasses)

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)


    #######################################
    ######## solvers Configuration ########
    #######################################


    chem = canteraChemMultiproc(th_file, ncells, nproc=10)
    transportFlux = AUSMPlusUpFlux()

    if nclasses > 0:
        sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
        solver = stcss.solver(grid, thermo, transportFlux, chem, sootSolver, bc, U0, savingInterval=10,
                            simdt=3.3e-7, savingPath='../../simulation_results/soot_model_validation', initialGridTemperature=T0)
    else:
        solver = s.solver(grid, thermo, transportFlux, chem, bc, U0, savingInterval=10, simdt=3.3e-7, savingPath='../../simulation_results/soot_model_validation', reactorModel='cst_rho', initialGridTemperature=T0)

    solver.solve(0.00033 * 4, 1000, 0.45, 'frozen_phi_2_20_flame.json')


    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation', 'frozen_phi_2_20_flame.json')
    if nclasses > 0:
        post_proc.animateField('T')
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
    else:
        post_proc.animateField('T')