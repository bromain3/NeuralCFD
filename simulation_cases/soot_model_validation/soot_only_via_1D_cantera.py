import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.soot_transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.growth.Lindset import Lindset as growthLIN
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.nucleation.Lindset import Lindset as nucLIN
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.coagulation.Fuchs import Fuchs
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.SootModel import SootModel
import NeuralCFD.solver.solver_soot_transport_soot_chemistry_subcycle as ssub
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
import cantera as ct

import torch
import numpy as np
import random


# TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change

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
    # f.radiation_enabled = True

    f.set_refine_criteria(ratio=refine_criterion[0], slope=refine_criterion[1], curve=refine_criterion[2])

    f.solve(loglevel=0, auto=True, refine_grid=True)

    return (f)


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

    # ISF laminar premixed flame 2 configuration (Xu et al., 1997; Menon et al., 2007)
    # Ethylene-air premixed burner stabilized flame
    # Références:
    # - Xu F. Combust Flame 1997;108(4):471–93
    # - Menon AV et al. Proc Combust Inst 2007;31(1):593–601
    equivalence_ratio = 2.34
    inflow_velocity = 6.73e-2  # m/s (converted from 6.73 cm/s)

    burner_diameter = 60.0 * 1e-3  # 60 mm burner diameter
    inflow_T = 298.0  # K - température ambiante (à ajuster selon les données expérimentales exactes)
    inflow_P = 101325.0  # Pa - pression atmosphérique

    #soot_d_min = 1.145853e-9 ; soot_d_max = 10_000.0e-9 # size used for LeungLindset model to ensure a nucleation of a particle at exactly 100 particles of carbon (if rho soot = 1800) as stated in the paper
    soot_d_min = 0.526715e-9 ; soot_d_max = 10_000.0e-9 # size used for Lindset model to ensure a nucleation of a particle at exactly 10 particles of carbon (if rho soot = 1800) as stated in the paper (Naphtalene)

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
    print("Computing 1D flame with cantera for ISF laminar premixed flame 2")
    print(f"Configuration: C2H4-Air, φ = {equivalence_ratio:.2f}, u = {inflow_velocity * 100:.2f} cm/s")

    gas = ct.Solution(th_file)
    gas.TP = inflow_T, inflow_P

    # Configuration avec éthylène (C2H4) au lieu de méthane
    gas.set_equivalence_ratio(equivalence_ratio, fuel='C2H4', oxidizer='O2:1.0, N2:3.76')

    # Extraction des fractions molaires pour vérification
    X_C2H4 = gas.X[gas.species_index('C2H4')]
    X_O2 = gas.X[gas.species_index('O2')]
    X_N2 = gas.X[gas.species_index('N2')]

    # Calcul du flux de masse à partir de la vitesse d'entrée
    # mdot = rho * u
    A = torch.pi * (burner_diameter / 2) ** 2
    mass_flux = gas.density * inflow_velocity

    print(f'Densité initiale du mélange: {gas.density:.4f} kg/m³')
    print(f'Flux de masse calculé: {mass_flux:.6f} kg/m²/s')
    print(f'Vitesse d\'entrée: {inflow_velocity * 100:.2f} cm/s')

    f = compute_burner_flame(gas, mass_flux, width=grid_length)

    # Visualisation des profils d'espèces clés
    plt.figure(figsize=(10, 6))
    plt.plot(f.grid * 1000, f.X[gas.species_names.index('C2H4')], label='C2H4')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Fraction molaire C2H4')
    plt.title('Profil de C2H4 - ISF laminar premixed flame 2')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(f.grid * 1000, f.X[gas.species_names.index('C2H2')], label='C2H2')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Fraction molaire C2H2')
    plt.title('Profil de C2H2 (précurseur de suies)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(f.grid * 1000, f.T, label='Temperature')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Température (K)')
    plt.title('Profil de température')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f'Vitesse calculée à partir du flux de masse: {mass_flux / gas.density:.4f} m/s')
    print(f"Richesse calculée (vérification): {gas.equivalence_ratio():.3f}")
    print(f"X_C2H4 = {X_C2H4:.6f}")
    print(f"X_O2   = {X_O2:.6f}")
    print(f"X_N2   = {X_N2:.6f}")
    print()

    ########################################################
    ######## creating the field from flame solution ########
    ########################################################
    ct_x_torch = torch.tensor(f.grid, device=device, dtype=dtype)
    ct_u_torch = torch.tensor(f.velocity, device=device, dtype=dtype)
    ct_T_torch = torch.tensor(f.T, device=device, dtype=dtype)
    ct_Y_torch = {spe: torch.tensor(f.Y[gas.species_names.index(spe)], device=device, dtype=dtype) for spe in
                  gas.species_names}

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
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state': U0[:, 0].flatten()}

    #######################################
    ######## solvers Configuration ########
    #######################################

    # sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    sootSolver = SootModel(nucLIN, oxiLL, growthLIN, FreeMolecular, grid, thermo)
    sootTransport = AUSMPlusUpFlux()  # rusanovFlux()

    solver = ssub.solver(grid, thermo, sootTransport, sootSolver, bc, U0, savingInterval=160, simdt=1e-5
                         , savingPath='../../simulation_results/soot_model_validation', initialGridTemperature=T0)

    solver.solve(50 * 0.0025, 30_000, 0.9, 'ISF_laminar_premixed_flame2_C2H4_phi2p34.json')

    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation',
                                  'ISF_laminar_premixed_flame2_C2H4_phi2p34.json')
    if nclasses > 0:
        post_proc.animateField('T')
        # post_proc.plotSpatialSootEvolution2D(-1)
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')