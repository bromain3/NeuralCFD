import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.POLENE_database_generator as db_gen
import NeuralCFD.solver.solver_transport_chemistry as s
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct

import torch
import random
#TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change
if __name__ == "__main__":

    def phi2Y(phi):
        MMCH4 = 16.04
        MMO2 = 31.999
        O2_N2 = 3.32

        mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
        Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
        return (Y)

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../input_data/gri30.yaml', device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0, 1, thermo, device=device, dtype=dtype)

    """
    flame_speed = 0.6105775028327183 * 1.05
    # rho0 = 1.0 + 0.0 * x
    p0 = torch.tensor([101325.0], device=device, dtype=dtype)
    u0 = torch.tensor([flame_speed], device=device, dtype=dtype)
    T0 = torch.tensor([400.0], device=device, dtype=dtype)
    Ys0 = torch.zeros((1, grid.nspecies), device=device, dtype=dtype)
    # Ys0[:, 0] = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.05**2)) + 0.25
    # Ys0[:, 1] = 1.0 - Ys0[:, 0]
    Ys0[:, thermo.thermoProp.species.index('CH4')] = 0.05483547715890419
    Ys0[:, thermo.thermoProp.species.index('N2')] = 0.7263764388501015
    Ys0[:, thermo.thermoProp.species.index('O2')] = 0.21878808399099442
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo).flatten()
    #bc = {'type': 'flame1d', 'inflow_state': U0, 'P_out': 101325.0}
    """
    gas = ct.Solution(thermo.thermoFilePath)
    bc = {'type': 'closed'}


    gas.TPY = 1700, 101325, {'CH4':0.05483547715890419, 'O2':0.21878808399099442, 'N2':0.7263764388501015}
    #gas.equilibrate('HP')
    # rho0 = 1.0 + 0.0 * x
    p0 = gas.P + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 =  gas.T + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
    #Ys0[:grid.ncells//2, :] = 0.0
    #Ys0[:grid.ncells//2, thermo.thermoProp.species.index('CH4')] = 0.05483547715890419
    #Ys0[:grid.ncells//2, thermo.thermoProp.species.index('N2')] = 0.7263764388501015
    #Ys0[:grid.ncells//2, thermo.thermoProp.species.index('O2')] = 0.21878808399099442

    #T0[:grid.ncells//2] = 400.0
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)




    #solver = s.solver(grid, thermo, bc, U0, savingInterval=10, savingPath='data', reactorModel='cst_p')
    #solver = snn.solver('../input_data/chem_models/export_PYTHON_const_pressure.pln' ,grid, thermo, bc, U0, savingInterval=10, savingPath='../data', reactorModel='cst_p')
    #solver = db_gen.solver(grid, thermo, bc, U0, savingPath='data', reactorModel='cst_rho')
    #solver.solve(0.0005, 10000, 1e-6, 'test.json')

    post_proc = pp.PostProcessing('../simulation_results/Flux_comparison', 'sod_choc_rus.json')
    #post_proc.plotSnapchot('T', 0)
    #post_proc.animateField('T')
    post_proc.plotVariables(['T', 'P'], 0)
    #post_proc.animateField('P')
    #post_proc.plotVariable('P', 0)
    #post_proc.animateField('Y_C2H2')