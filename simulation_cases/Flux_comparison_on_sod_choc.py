import os.path

import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport as solv_trans
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlus import AUSMPlusFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc


import torch

if __name__ == "__main__":

    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    #torch.cuda.memory._set_allocator_settings()
    #torch.cuda.memory._get_current_allocator()
    #torch.cuda.memory.max_memory_reserved()
    #torch.cuda.memory.max_memory_cached()
    #torch.cuda.memory.max_memory_allocated()

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

    saving_path = '../simulation_results/Flux_comparison'

    transportFlux = rusanovFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    solver.solve(0.005, 100000, 0.50, 'sod_choc_rus.json')

    transportFlux = AUSMFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM.json')

    transportFlux = AUSMPlusFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM+.json')

    transportFlux = AUSMPlusUpFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM+up.json')

    post_proc = pp.PostProcessing('', '')
    #post_proc.animateField('rho')
    post_proc.animateFields([os.path.join(saving_path, 'sod_choc_rus.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM+.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM+up.json')], 'rho')