import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.POLENE_database_generator as db_gen
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.soot.oneWayHACA_CPU import oneWayHACA_CPU
from NeuralCFD.soot.oneWayHACA_GPU import oneWayHACA_GPU
from NeuralCFD.chemestry.canteraChem import canteraChem
from NeuralCFD.chemestry.canteraChemMultiproc import canteraChemMultiproc
import NeuralCFD.solver.solver_transport_chemistry as s
import NeuralCFD.solver.solver_transport as st
import NeuralCFD.solver.solver_transport_soot as sts
import NeuralCFD.solver.solver_transport_chemistry_soot as stcs
import NeuralCFD.solver.solver_transport_chemistry_soot_subcycle as stcss

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

    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    ncells = 1000
    nclasses = 20
    th_file = r'../../input_data/gri30.yaml'
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(10.0e-3 / 2, ncells, thermo, device=device, dtype=dtype, sectionalParams={"dmin": 1.0e-9, "dmax": 100.0e-9, "nclasses": nclasses, 'spacing': 'log'})

    bc = {'type': 'symetric'}
    gas = ct.Solution(thermo.thermoFilePath)

    gas.Y = phi2Y(1.0)

    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 = 650.0 + (2000.0 - 650.0) * torch.exp(-(grid.x / (1.8 * 4 * 2.5e-4)) ** 2)  # hot spot in center
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)


    chem = canteraChemMultiproc(th_file, ncells, nproc=10)
    #chem = canteraChem(th_file, ncells)
    transportFlux = rusanovFlux()
    #sootSolver = oneWayHACA_CPU(grid, thermo)
    sootSolver = oneWayHACA_GPU(grid, thermo)


    #solver = stcs.solver(grid, thermo, transportFlux, chem, sootSolver, bc, U0, savingInterval=10, savingPath='../simulation_results/sooting_flamme', initialGridTemperature=T0)
    solver = stcss.solver(grid, thermo, transportFlux, chem, sootSolver, bc, U0, savingInterval=10, simdt=1.0e-7, savingPath='../simulation_results/sooting_flamme', initialGridTemperature=T0)

    #solver.solve(0.0005, 100_000, 0.95, 'first_test.json')


    post_proc = pp.PostProcessing('../simulation_results/sooting_flamme', 'first_test.json')
    #post_proc.plotSnapchot('N_0', -1)
    #post_proc.animateField('N_10')
    #post_proc.plotSpatialSootEvolution2D(-1)
    #post_proc.plotSpatialSootEvolution3D(-1, y_axis='log_size')
    #post_proc.plotTemporalSootEvolution2D(0)
    #post_proc.plotTemporalSootEvolution3D(0, y_axis='log_size')
    post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
    #for i in range(nclasses):
        #post_proc.plotSnapchot(f'N_{i}', -1)
        #post_proc.animateField(f'N_{i}')
    #post_proc.plotVariables(['T', 'P'], 0)
    #post_proc.animateField('P')
    #post_proc.plotVariable('P', 0)
    #post_proc.animateField('T')