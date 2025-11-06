import numpy as np

import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.thermo.vectorized_thermo as th
import NeuralCFD.solver.solver_transport_rusanov_cantera_chemestry as ctsolv
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_rusanov_network_chemestry as neuralsolv
import NeuralCFD.thermo.neuralThermo as nt

import cantera as ct
import torch




dt = 3.0e-7

# TODO: investigate the strange value of gas.P after setting gas.TP
def phi2Y(phi):
    MMCH4 = 16.04
    MMO2 = 31.999
    O2_N2 = 3.32

    mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
    Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
    return (Y)

def full_cantera(Tini, phiini):
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml',
                                                       device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, 101325.0
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    solver = ctsolv.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='map_full_cantera_200', reactorModel='cst_p')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')


def full_network(Tini, phiini):
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml',device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, gas.P
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    print(T0, p0, gas.P)

    solver = neuralsolv.solver(r'vectorized_thermo_network_chemestry_gpu/chem_models/export_PYTHON_full_chem.pln', grid, thermo, bc, U0, savingInterval=1, savingPath='map_full_network_200')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')

def reduced_network_NASA7(Tini, phiini):
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml',device=device, dtype=dtype, reducedSpeciesSet=["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "CH2", "CH3", "CH4", "CO", "CO2", "C2H2", "N2", "bin"])
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, gas.P
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        if spe in thermo.thermoProp.species:
            Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
    if ~torch.isclose(torch.sum(Ys0, dim=1), torch.tensor(1.0, dtype=dtype, device=device)).any():
        print(~torch.isclose(torch.sum(Ys0, dim=1), torch.tensor(1.0, dtype=dtype, device=device)))
        raise ValueError('mass fractions does not add up to 1 in at least one cell')

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)


    solver = neuralsolv.solver(r'vectorized_thermo_network_chemestry_gpu/chem_models/export_PYTHON_chimie_reduite.pln', grid, thermo, bc, U0, savingInterval=1, savingPath='map_reduced_network_NASA7_200')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')

def SAR_28(Tini, phiini):
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/chem_models/red_SAR_C2H2_28_spe/Red_mech/1_C1_GRI30.yaml',
                                                       device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, 101325.0
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    solver = ctsolv.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='map_SAR_28_200', reactorModel='cst_p')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')

def SAR_15(Tini, phiini):
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/chem_models/red_SAR_C2H2_15_spe/Red_mech/1_C1_GRI30.yaml',
                                                       device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, 101325.0
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    solver = ctsolv.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='map_SAR_15_200', reactorModel='cst_p')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')

def DRGEP_35(Tini, phiini):
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/chem_models/red_DRGEP_C2H2_35_spe/Red_mech/1_C1_GRI30.yaml',
                                                       device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, 101325.0
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    solver = ctsolv.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='map_DRGEP_35_200', reactorModel='cst_p')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')


def DRGEP_24(Tini, phiini):
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/chem_models/red_DRGEP_C2H2_24_spe/Red_mech/1_C1_GRI30.yaml',
                                                       device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}

    # gas.equilibrate('HP')
    gas = ct.Solution(thermo.thermoFilePath)
    gas.TP = Tini, 101325.0
    gas.Y = phi2Y(phiini)
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = gas.T + 0.0 * grid.x

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    solver = ctsolv.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='map_DRGEP_24_200', reactorModel='cst_p')
    solver.solve(0.001, 10000000, dt, f'T_{Tini}_phi_{phiini}.json')

if __name__ == '__main__':
    from joblib import Parallel, delayed

    n_jobs = 90

    phiini = np.linspace(0.73, 1.47, 100)
    Tini = np.linspace(1620, 1780, 100)


    args = []
    for T in Tini:
        for phi in phiini:
            args.append((T, phi))


    model = full_cantera

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    model = SAR_15

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    model = DRGEP_35

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    model = SAR_28

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    model = DRGEP_24

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    """
    model = full_network

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)

    model = reduced_network_NASA7

    Parallel(n_jobs=n_jobs, backend='loky')(delayed(model)(T, phi) for T, phi in args)
    """