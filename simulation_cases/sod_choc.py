import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_rusanov as solv_rus
import NeuralCFD.solver.solver_transport_AUSM as solv_AUSM
import NeuralCFD.solver.solver_transport_AUSMPlus as solv_AUSMPlus
import NeuralCFD.solver.solver_transport_AUSMPlusUp as solv_AUSMPlusUp
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc


import torch

if __name__ == "__main__":

    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../input_data/gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)


    bc = {'type': 'closed'}

    rho0 = 1.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    p0 = 101325.0 + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    Ys0[:, 0] = 1.0

    rho0[len(grid.x)//2:] = 0.125
    p0[len(grid.x) // 2:] = 0.1 * 101325.0

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    T0 = p0 / Rspec0 / rho0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)
    print(p0[0], p0[-1])
    print(T0[0], T0[-1])


    solver = solv_rus.solver(grid, thermo, bc, U0, savingInterval=10, savingPath='../data')

    solver.solve(0.005, 100000, 0.5, 'sod_choc_custom_S.json')

    post_proc = pp.PostProcessing('../data', 'sod_choc_custom_S.json')
    #post_proc.animateField('rho')
    post_proc.animateFields(['../data/sod_choc_rus.json', '../data/sod_choc_AUSM.json', '../data/sod_choc_AUSM+.json', '../data/sod_choc_AUSM+up.json', '../data/sod_choc_custom_S.json'], 'rho')

    #5328Mo vram and 0.02205 s of computation time per step before GPU memory optimisation