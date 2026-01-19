import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_chemistry as solv
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.chemestry.canteraChem import canteraChem
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    th_file = r'../../input_data/gri30.yaml'
    ncells = 1

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    chem = canteraChem(th_file, ncells)

    grid = NeuralCFD.grid.grid.grid(1.0, ncells, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}
    gas = ct.Solution(thermo.thermoFilePath)

    P = 101325.0
    T = 1700.0
    u = 0.0

    gas.TPY = T, P, {'CH4': 0.05483547715890419, 'O2': 0.21878808399099442, 'N2': 0.7263764388501015}

    T0 = T + 0.0 * grid.x
    u0 = u + 0.0 * grid.x
    p0 = P + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    transportFlux = rusanovFlux()
    solver = solv.solver(grid, thermo, transportFlux, chem, bc, U0, maxTimestep=1e-7, savingInterval=10,
                         savingPath='../../simulation_results/homogeneous_reactors', reactorModel='cst_p',
                         initialGridTemperature=T0)
    solver.solve(0.0005, 100_000, 0.95, 'cst_p.json')

    post_proc = pp.PostProcessing('../../simulation_results/homogeneous_reactors', 'cst_p.json')
    post_proc.plotVariables('T', 0)
    post_proc.plotVariables('P', 0)
    post_proc.plotVariables('u', 0)
    post_proc.animateField('T')