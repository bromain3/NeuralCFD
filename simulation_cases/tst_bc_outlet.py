import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_rusanov as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc


import torch

if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)


    bc = {'type': 'outlets_nonreflective', 'P_out': 101325.0, 'beta': 0.5}

    rho0 = 0.13 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    p0 = 101325.0 + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    Ys0[:, 0] = 1.0

    rho0[len(grid.x)//2:] = 0.125
    #p0[len(grid.x) // 2:] = 0.1 * 101325.0

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    T0 = p0 / Rspec0 / rho0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)


    solver = s.solver(grid, thermo, bc, U0, savingInterval=5, savingPath='test_outlet')

    solver.solve(0.005, 100000, 0.95, 'sim.json')

    post_proc = pp.PostProcessing('test_outlet', 'sim.json')
    post_proc.animateField('P')