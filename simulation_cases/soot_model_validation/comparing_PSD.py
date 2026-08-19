import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.soot_transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.growth.Lindset import Lindset as growthLIN
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.nucleation.Lindset import Lindset as nucLIN
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.coagulation.Fuchs_vect import Fuchs_vect
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular_vect import FreeMolecular_vect
from NeuralCFD.soot_chemistry.SootModel import SootModel
import NeuralCFD.solver.solver_soot_transport_soot_chemistry_subcycle as ssub
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
import cantera as ct

import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def compute_burner_flame(gas, mdot, refine_criterion=(3.0, 0.05, 0.1), width=0.03):
    """
    Compute burner-stabilized flame with Cantera

    Args:
        gas: Cantera Solution object
        mdot: Mass flux (kg/m²/s)
        refine_criterion: Grid refinement criteria (ratio, slope, curve)
        width: Domain width (m)
    """
    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = mdot

    f.set_refine_criteria(ratio=refine_criterion[0], slope=refine_criterion[1], curve=refine_criterion[2])

    f.solve(loglevel=0, auto=True, refine_grid=True)

    return f


if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cuda:1'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    ncells = 5000
    grid_length = 25.0e-3  # 25 mm domain
    nclasses = 40

    # Soot model parameters
    # soot_d_min = 1.145853e-9 ; soot_d_max = 10_000.0e-9 # size used for LeungLindset model to ensure a nucleation of a particle at exactly 100 particles of carbon (if rho soot = 1800) as stated in the paper
    soot_d_min = 0.526715e-9; soot_d_max = 10_000.0e-9  # size used for Lindset model to ensure a nucleation of a particle at exactly 10 particles of carbon (if rho soot = 1800) as stated in the paper (Naphtalene)

    th_file = r'../../input_data/gri30.yaml'

    ################################################################################################################################################################
    #### BURNER-STABILIZED FLAME EXPERIMENTAL CONDITIONS                                                                                                        ####
    #### Article: On evolution of particle size distribution functions of incipient soot in premixed ethylene–oxygen–argon flames                               ####
    #### Composition: 16.3% C2H4, 23.7% O2, 60% Ar (fixed)                                                                                                      ####
    #### φ = 2.07 (fixed for all flames)                                                                                                                        ####
    #### Variable: Cold gas velocity (controls flame temperature)                                                                                               ####
    ################################################################################################################################################################

    # Burner temperature (typical for water-cooled burners)
    # The article mentions Type K thermocouple at bottom of porous plug
    # Typical range: 320-360 K for water-cooled burners
    burner_temperature = 300.0  # K - STP temperature (can be adjusted to 340-360 K if needed)

    # Fixed composition for all flames (Table 1)
    X_C2H4 = 0.163  # Mole fraction of ethylene
    X_O2 = 0.237  # Mole fraction of oxygen
    X_Ar = 0.600  # Mole fraction of argon (balance)
    equivalence_ratio = 2.07  # Fixed equivalence ratio

    # Select which flame to simulate (C1 to C6)
    # Change this to simulate different flames from Table 1
    FLAME_SELECTION = 'C4'  # Options: 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'

    # Flame conditions from Table 1
    flame_conditions = {
        'C1': {'v0_cm_s': 13.0, 'Tf_max': 1898, 'Tf_std': 50},
        'C2': {'v0_cm_s': 10.0, 'Tf_max': 1805, 'Tf_std': 50},
        'C3': {'v0_cm_s': 8.0, 'Tf_max': 1736, 'Tf_std': 50},
        'C4': {'v0_cm_s': 6.53, 'Tf_max': 1710, 'Tf_std': 50},
        'C5': {'v0_cm_s': 5.5, 'Tf_max': 1660, 'Tf_std': 50},
        'C6': {'v0_cm_s': 4.5, 'Tf_max': 1610, 'Tf_std': 50}  # Note: unstable flame
    }

    # Get selected flame conditions
    selected_flame = flame_conditions[FLAME_SELECTION]
    inflow_velocity = selected_flame['v0_cm_s'] * 1e-2  # Convert cm/s to m/s
    expected_Tf_max = selected_flame['Tf_max']

    # Burner specifications from the article
    burner_diameter = 3.0 * 2.54e-2  # 3 inches outer diameter = 7.62 cm
    shroud_N2_flow = 55.0  # L/min nitrogen shroud flow
    inflow_P = 101325.0  # Pa - atmospheric pressure

    #############################################################
    ########## Display configuration information     ############
    #############################################################
    print()
    print("=" * 80)
    print(f"BURNER-STABILIZED FLAME CONFIGURATION - Flame {FLAME_SELECTION}")
    print("=" * 80)
    print(f'Experimental conditions from article:')
    print(f'  - Composition (fixed): {X_C2H4 * 100:.1f}% C2H4, {X_O2 * 100:.1f}% O2, {X_Ar * 100:.1f}% Ar')
    print(f'  - Equivalence ratio φ: {equivalence_ratio:.2f} (fixed)')
    print(f'  - Cold gas velocity v0: {inflow_velocity * 100:.2f} cm/s')
    print(f'  - Expected max temperature: {expected_Tf_max} ± {selected_flame["Tf_std"]} K')
    print(f'  - Burner diameter: {burner_diameter * 100:.2f} cm (3 inch)')
    print(f'  - Shroud N2 flow: {shroud_N2_flow} L/min')
    print(f'  - Burner temperature: {burner_temperature:.0f} K')
    print(f'  - Pressure: {inflow_P / 101325:.2f} atm')
    print()
    print(f'Grid configuration:')
    print(f'  - Domain length: {grid_length * 1000:.1f} mm')
    print(f'  - Number of cells: {ncells}')
    print(f'  - Cell size: {grid_length * 1000 / ncells:.4f} mm')
    print()
    print(f'Soot model:')
    print(f'  - Smallest particle diameter: {soot_d_min:.5e} m')
    print(f'  - Largest particle diameter: {soot_d_max:.5e} m')
    print(f'  - Number of size classes: {nclasses}')
    print("=" * 80)
    print()

    #####################################
    ########## Object creation ##########
    #####################################
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    print(f'Thermo created with file: {th_file}')
    print()

    grid = NeuralCFD.grid.grid.grid(grid_length, ncells, thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max,
                                                     "nclasses": nclasses, 'spacing': 'log'})

    print(f'Grid created with sectional model enabled.')
    print()

    ######################################
    ###### Flame computation      ########
    ######################################

    print("Computing 1D burner-stabilized flame with Cantera")
    print(f"Configuration: Flame {FLAME_SELECTION} - C2H4/O2/Ar")
    print()

    # Create Cantera gas object
    gas = ct.Solution(th_file)

    # CRITICAL FIX: Set composition FIRST, then temperature and pressure
    # If you set TP before composition, Cantera recalculates pressure incorrectly
    composition_dict = {
        'C2H4': X_C2H4,
        'O2': X_O2,
        'AR': X_Ar  # Cantera uses 'AR' for argon
    }
    gas.X = composition_dict

    # NOW set temperature and pressure AFTER composition
    gas.TP = burner_temperature, inflow_P

    # Verify the equivalence ratio
    # For C2H4 + 3O2 -> 2CO2 + 2H2O (stoichiometric)
    phi_verify = (X_C2H4 / X_O2) / (1.0 / 3.0)

    print(f'Initial mixture properties:')
    print(f'  - Temperature: {gas.T:.1f} K')
    print(f'  - Pressure: {gas.P / 101325:.3f} atm')
    print(f'  - Density: {gas.density:.4f} kg/m³')
    print(f'  - X_C2H4: {gas.X[gas.species_index("C2H4")]:.4f}')
    print(f'  - X_O2: {gas.X[gas.species_index("O2")]:.4f}')
    print(f'  - X_AR: {gas.X[gas.species_index("AR")]:.4f}')
    print(f'  - Verified equivalence ratio: {phi_verify:.3f}')
    print()

    # Calculate mass flux from velocity
    # IMPORTANT: Use the density AFTER setting composition and TP
    mass_flux = gas.density * inflow_velocity

    print(f'Mass flux: {mass_flux:.6f} kg/m²/s')
    print(f'Inlet velocity: {inflow_velocity * 100:.2f} cm/s')
    print(f'Calculated velocity check: {mass_flux / gas.density * 100:.2f} cm/s')
    print()

    # Solve the burner-stabilized flame
    print("Solving burner-stabilized flame...")
    f = compute_burner_flame(gas, mass_flux, width=grid_length)
    print(f"Flame solved successfully!")
    print(f"  - Grid points: {len(f.grid)}")
    print(f"  - Maximum temperature: {f.T.max():.1f} K")
    print(f"  - Expected Tf,max: {expected_Tf_max} ± {selected_flame['Tf_std']} K")
    print(f"  - Temperature difference: {f.T.max() - expected_Tf_max:.1f} K")
    print(f"  - Domain length: {f.grid[-1] * 1000:.2f} mm")
    print()

    # Check if temperature is within expected range
    if abs(f.T.max() - expected_Tf_max) > selected_flame['Tf_std']:
        print("WARNING: Maximum flame temperature is outside expected range!")
        print("         Consider adjusting burner_temperature.")
        print(f"         Try burner_temperature = {burner_temperature + 40:.0f} K to increase Tf,max")
    else:
        print("✓ Maximum flame temperature is within expected range.")
    print()

    # Visualize key species profiles
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # C2H4 profile
    axes[0, 0].plot(f.grid * 1000, f.X[gas.species_names.index('C2H4')], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[0, 0].set_ylabel('Mole fraction C2H4', fontsize=11)
    axes[0, 0].set_title('Ethylene (C2H4) Profile', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, grid_length * 1000])

    # C2H2 profile (soot precursor)
    axes[0, 1].plot(f.grid * 1000, f.X[gas.species_names.index('C2H2')], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[0, 1].set_ylabel('Mole fraction C2H2', fontsize=11)
    axes[0, 1].set_title('Acetylene (C2H2) - Soot Precursor', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, grid_length * 1000])

    # Temperature profile
    axes[0, 2].plot(f.grid * 1000, f.T, 'orange', linewidth=2)
    axes[0, 2].axhline(y=expected_Tf_max, color='red', linestyle='--',
                       label=f'Expected Tf,max = {expected_Tf_max} K', linewidth=1.5)
    axes[0, 2].axhspan(expected_Tf_max - selected_flame['Tf_std'],
                       expected_Tf_max + selected_flame['Tf_std'],
                       alpha=0.2, color='red', label='±50 K range')
    axes[0, 2].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[0, 2].set_ylabel('Temperature (K)', fontsize=11)
    axes[0, 2].set_title(f'Temperature Profile - Flame {FLAME_SELECTION}',
                         fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].set_xlim([0, grid_length * 1000])

    # Velocity profile
    axes[1, 0].plot(f.grid * 1000, f.velocity * 100, 'g-', linewidth=2)
    axes[1, 0].axhline(y=inflow_velocity * 100, color='gray', linestyle='--',
                       label=f'Inlet v0 = {inflow_velocity * 100:.2f} cm/s', linewidth=1.5)
    axes[1, 0].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[1, 0].set_ylabel('Velocity (cm/s)', fontsize=11)
    axes[1, 0].set_title('Velocity Profile', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xlim([0, grid_length * 1000])

    # OH profile (flame marker)
    axes[1, 1].plot(f.grid * 1000, f.X[gas.species_names.index('OH')], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[1, 1].set_ylabel('Mole fraction OH', fontsize=11)
    axes[1, 1].set_title('OH Radical Profile', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, grid_length * 1000])

    # Density profile
    axes[1, 2].plot(f.grid * 1000, f.density, 'c-', linewidth=2)
    axes[1, 2].set_xlabel('Height above burner (mm)', fontsize=11)
    axes[1, 2].set_ylabel('Density (kg/m³)', fontsize=11)
    axes[1, 2].set_title('Density Profile', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim([0, grid_length * 1000])

    plt.suptitle(f'Burner-Stabilized Flame {FLAME_SELECTION} - C2H4/O2/Ar (φ={equivalence_ratio:.2f})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    # Additional verification output
    print()
    print("=" * 80)
    print("VERIFICATION OF FLAME CONDITIONS")
    print("=" * 80)
    print(f'Flame: {FLAME_SELECTION}')
    print(f'Inlet velocity (from flame solution): {f.velocity[0] * 100:.2f} cm/s')
    print(f'Target inlet velocity: {inflow_velocity * 100:.2f} cm/s')
    print(f'Velocity ratio: {f.velocity[0] / inflow_velocity:.3f}')
    print(f'Maximum flame temperature: {f.T.max():.1f} K')
    print(f'Expected temperature: {expected_Tf_max} ± {selected_flame["Tf_std"]} K')
    print(f'Maximum C2H2 mole fraction: {f.X[gas.species_names.index("C2H2")].max():.6f}')
    print(f'Maximum OH mole fraction: {f.X[gas.species_names.index("OH")].max():.6f}')
    print(f'Location of max T: {f.grid[np.argmax(f.T)] * 1000:.2f} mm')
    print(f'Location of max C2H2: {f.grid[np.argmax(f.X[gas.species_names.index("C2H2")])] * 1000:.2f} mm')
    print("=" * 80)
    print()

    ########################################################
    ######## Creating the field from flame solution ########
    ########################################################
    print("Interpolating Cantera solution onto computational grid...")

    ct_x_torch = torch.tensor(f.grid, device=device, dtype=dtype)
    ct_u_torch = torch.tensor(f.velocity, device=device, dtype=dtype)
    ct_T_torch = torch.tensor(f.T, device=device, dtype=dtype)
    ct_Y_torch = {spe: torch.tensor(f.Y[gas.species_names.index(spe)], device=device, dtype=dtype)
                  for spe in gas.species_names}

    u0 = torch_interp(grid.x, ct_x_torch, ct_u_torch)
    T0 = torch_interp(grid.x, ct_x_torch, ct_T_torch)
    Y0 = {spe: torch_interp(grid.x, ct_x_torch, ct_Y_torch[spe]) for spe in gas.species_names}
    Y0 = torch.stack([Y0[spe] for spe in gas.species_names]).T
    p0 = inflow_P + 0.0 * grid.x
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Y0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Y0, ns0, thermo)

    print("Interpolation complete.")
    print()

    #####################################
    ######## Boundary conditions ########
    #####################################
    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state': U0[:, 0].flatten()}

    #######################################
    ######## Solvers Configuration ########
    #######################################
    print("Configuring soot model...")
    print(f"  - Nucleation: Lindset model")
    print(f"  - Oxidation: LeungLindset model")
    print(f"  - Growth: Lindset model")
    print(f"  - Coagulation: Free molecular (vectorized)")
    print()

    sootSolver = SootModel(nucLIN, oxiLL, growthLIN, FreeMolecular_vect, grid, thermo)
    sootTransport = AUSMPlusUpFlux()

    solver = ssub.solver(grid, thermo, sootTransport, sootSolver, bc, U0,
                         savingInterval=700,
                         simdt=1e-5,
                         savingPath='../../simulation_results/soot_model_validation',
                         initialGridTemperature=T0)


    solver.solve(0.38, 50_000, 0.9,
                 f'Flame_{FLAME_SELECTION}_C2H4_O2_Ar_phi2p07.json')


    #######################################
    ######## Post-processing ##############
    #######################################
    post_proc = pp.PostProcessing('../../simulation_results/soot_model_validation',
                                  f'Flame_{FLAME_SELECTION}_C2H4_O2_Ar_phi2p07.json')

    if nclasses > 0:
        post_proc.animateField('T')
        post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        post_proc.plotSootVolumeFraction(-1)
        post_proc.animateSootVolumeFraction()
    else:
        post_proc.animateField('T')
