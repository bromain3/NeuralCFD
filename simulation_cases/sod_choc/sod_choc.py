import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport as solv_trans
from NeuralCFD.interpolation.MUSCL import MUSCL, minmod, vanleer
#from NeuralCFD.interpolation.WENO import WENO5
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlus import AUSMPlusFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc


import torch

if __name__ == "__main__":

    device = 'cuda:1'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    #torch.cuda.memory._set_allocator_settings()
    #torch.cuda.memory._get_current_allocator()
    #torch.cuda.memory.max_memory_reserved()
    #torch.cuda.memory.max_memory_cached()
    #torch.cuda.memory.max_memory_allocated()

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../../input_data/gri30.yaml', device=device, dtype=dtype)

    interpMethode = MUSCL(minmod)
    #interpMethode = WENO5()
    grid = NeuralCFD.grid.grid.grid(1.0, 2000, thermo, interpolationMethode=interpMethode, device=device, dtype=dtype)
    #grid = NeuralCFD.grid.grid.grid(1.0, 2000, thermo, device=device, dtype=dtype)  # max cells number on 64Go VRAM : ~4_700_000

    bc = {'type': 'closed'}








    P = 101325.0 * 20

    rho0 = 1.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    p0 = P + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    Ys0[:, 0] = 1.0

    rho0[len(grid.x)//2:] = 0.125
    p0[len(grid.x) // 2:] = 0.1 * P

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    T0 = p0 / Rspec0 / rho0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)
    print(T0[0], T0[-1])
    print(p0[0], p0[-1])












    transportFlux = rusanovFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath='../../simulation_results/sod_choc', initialGridTemperature=T0)
    solver.solve(0.001, 100000, 0.30, 'sod_choc_rus.json')


    post_proc = pp.PostProcessing(r'../../simulation_results/sod_choc', 'sod_choc_rus.json')
    post_proc.animateField('rho')
    #post_proc.animateFields(['../../simulation_results/Flux_comparison/sod_choc_rus.json', '../../simulation_results/Flux_comparison/sod_choc_AUSM.json', '../../simulation_results/Flux_comparison/sod_choc_AUSM+.json', '../../simulation_results/Flux_comparison/sod_choc_AUSM+up.json'], 'rho')