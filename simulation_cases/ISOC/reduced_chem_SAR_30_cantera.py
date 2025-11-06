import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.thermo.vectorized_thermo as th
import NeuralCFD.solver.solver_transport_rusanov_cantera_chemestry as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct

import torch
#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)


    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'vectorized_thermo_network_chemestry_gpu/chem_models/red_SAR_C2H2_30_spe/Red_mech/1_C1_GRI30.yaml', device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)

    bc = {'type': 'closed'}




    #gas.equilibrate('HP')
    # rho0 = 1.0 + 0.0 * x
    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 =  1700.0 + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)

    Ys0[:, thermo.thermoProp.species.index('CH4')] = 0.05483547715890419 + 0.0 * grid.x
    Ys0[:, thermo.thermoProp.species.index('O2')] = 0.21878808399099442 + 0.0 * grid.x
    Ys0[:, thermo.thermoProp.species.index('N2')] = 0.7263764388501015 + 0.0 * grid.x


    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)




    solver = s.solver(grid, thermo, bc, U0, savingInterval=1, savingPath='data', reactorModel='cst_p')
    solver.solve(0.0005, 10000, 1e-6, 'auto_allumage_SAR_30_cantera.json')

    post_proc = pp.PostProcessing('data', 'auto_allumage_SAR_30_cantera.json')
    post_proc.animateField('T')
    post_proc.plotVariable('T', 0)
    post_proc.animateField('P')
    post_proc.plotVariable('P', 0)
    post_proc.animateField('Y_C2H2')
    post_proc.plotVariable('Y_C2H2', 0)

    # simulation with saving every timestep on multithreaded cpu of personal computer. no gpu
    # physical time : 0.0005 s
    # total sim time : 129.1554028988 s
    # Total transport time: 10.57394 s
    # Mean transport time per step: 0.02115 s
    # Total chemistry time: 118.09038 s
    # Mean chemistry time per step: 0.23618 s