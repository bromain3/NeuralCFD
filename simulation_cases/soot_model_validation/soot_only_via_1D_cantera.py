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
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
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


def compute_burner_flame(gas, mdot, refine_criterion=(3.0, 0.05, 0.1), width=0.03):
    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = mdot
    #f.radiation_enabled = True

    f.set_refine_criteria(ratio=refine_criterion[0], slope=refine_criterion[1], curve=refine_criterion[2])

    f.solve(loglevel=0, auto=True, refine_grid=True)

    return (f)

if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    ncells = 5000
    grid_length = 25.0e-3
    nclasses = 40

    th_file = r'../../input_data/gri30.yaml'

    ox_inflow_mdot = 139.2 * 1e-6
    fuel_inflow_mdot = 40.1 * 1e-6
    burner_diameter = 60.0 * 1e-3
    inflow_T = 294.0
    inflow_P = 101325.0

    soot_d_min = 1.0e-9
    soot_d_max = 100.0e-9

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

    grid = NeuralCFD.grid.grid.grid(grid_length, ncells, thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses,
                                                     'spacing': 'log'})

    print()
    if nclasses > 0:
        print(
            f'Grid created with sectional model enabled. Sectional model params are d_min={soot_d_min} and d_max={soot_d_max} with nclasses={nclasses}')
    else:
        print(f'Grid created with sectional model disabled.')
    print()

    ######################################
    ###### flamme speed computation ######
    ######################################
    import matplotlib.pyplot as plt
    print()
    print("Computing 1D flame with cantera for having flame field")
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TPY = inflow_T, inflow_P, {'CH4': fuel_inflow_mdot, 'O2': ox_inflow_mdot}

    mdot_tot = (ox_inflow_mdot + fuel_inflow_mdot) / (torch.pi * (burner_diameter / 2)**2)
    f = compute_burner_flame(gas, mdot_tot, width=grid_length)
    plt.show()
    print()

    ########################################################
    ######## creating the field from flame solution ########
    ########################################################
    ct_x_torch = torch.tensor(f.grid, device=device, dtype=dtype)
    ct_u_torch = torch.tensor(f.velocity, device=device, dtype=dtype)
    ct_T_torch = torch.tensor(f.T, device=device, dtype=dtype)
    ct_Y_torch = {spe:torch.tensor(f.Y[gas.species_names.index(spe)], device=device, dtype=dtype) for spe in gas.species_names}

    u0 = torch_interp(grid.x, ct_x_torch, ct_u_torch)
    T0 = torch_interp(grid.x, ct_x_torch, ct_T_torch)
    Y0 = {spe: torch_interp(grid.x, ct_x_torch, ct_Y_torch[spe]) for spe in gas.species_names}
    Y0 = torch.stack([Y0[spe] for spe in gas.species_names]).T
    p0 = inflow_P + 0.0 * grid.x
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Y0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Y0, ns0, thermo)



    #####################################
    ######## boundary conditions ########
    #####################################
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U0[:, 0].flatten()}


    #######################################
    ######## solvers Configuration ########
    #######################################



    sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    sootTransport = AUSMPlusUpFlux() #rusanovFlux()

    solver = s.solver(grid, thermo, sootTransport, sootSolver, bc, U0, savingInterval=100
                      , savingPath='../../simulation_results/soot_model_validation', initialGridTemperature=T0)

    solver.solve(0.00033, 1000, 0.95, 'F_XU_comparison_cantera_soot.json')


    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation', 'F_XU_comparison_cantera_soot.json')
    if nclasses > 0:
        post_proc.animateField('T')
        post_proc.plotSpatialSootEvolution2D(-1)
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')