import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.POLENE_database_generator as db_gen
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.oneWay_Leung_Lindset import oneWay_Leung_Lindset
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.coagulation.Fuchs import Fuchs
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.SootModel import SootModel
from NeuralCFD.chemestry.canteraChem import canteraChem
from NeuralCFD.chemestry.canteraChemMultiproc import canteraChemMultiproc
import NeuralCFD.solver.solver_transport_chemistry as s
import NeuralCFD.solver.solver_transport as st
import NeuralCFD.solver.solver_transport_soot as sts
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

    ncells = 10
    nclasses = 10
    th_file = r'../input_data/gri30.yaml'
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    grid = NeuralCFD.grid.grid.grid(1.0e-6, ncells, thermo, device=device, dtype=dtype, sectionalParams={"dmin": 1.0e-9, "dmax": 100.0e-9, "nclasses": nclasses, 'spacing': 'log'})

    gas = ct.Solution(thermo.thermoFilePath)
    bc = {'type': 'closed'}

    # O2 oxydation
    #gas.TPY = 1700, 101325, {'CH4':0.05483547715890419, 'O2':0.21878808399099442, 'N2':0.7263764388501015}
    # OH oxydation
    #gas.TPY = 1700, 101325, {'CH4': 0.05483547715890419, 'OH': 0.0021878808399099442, 'N2': 0.7263764388501015}
    #neutral gas (pur coagulation)
    gas.TPY = 1700, 101325, {'CH4': 0.05483547715890419, 'N2': 0.7263764388501015}
    # nucleation and surface growth and nucleation
    #gas.TPY = 2500, 101325, {'CH4': 0.05483547715890419, 'C2H2': 0.21878808399099442, 'N2': 0.7263764388501015}
    # all processes
    #gas.TPY = 1000, 101325, {'CH4': 0.05483547715890419,'O2':0.21878808399099442, 'OH': 0.021878808399099442,  'C2H2': 0.061878808399099442, 'N2': 0.7263764388501015}
    #gas.equilibrate('HP')
    # rho0 = 1.0 + 0.0 * x
    p0 = gas.P + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 =  gas.T + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    ns0[:, 5] = 1.0e20
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)



    transportFlux = rusanovFlux()
    #transportFlux = AUSMPlusUpFlux()
    #chem = canteraChem(th_file, ncells)
    #chem = canteraChemMultiproc(th_file, ncells, nproc=10)
    #solver = s.solver(grid, thermo, transportFlux, chem, bc, U0, savingInterval=10, savingPath='data', reactorModel='cst_rho', initialGridTemperature=T0)
    #sootSolver = oneWay_Leung_Lindset(grid, thermo)
    sootSolver = SootModel(nucLL, oxiLL, growthLL, FreeMolecular, grid, thermo)

    #solver = st.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath='../simulation_results/test', initialGridTemperature=T0)
    solver = sts.solver(grid, thermo, transportFlux, sootSolver, bc, U0, savingInterval=1, savingPath='../simulation_results/test', initialGridTemperature=T0)
    #solver = snn.solver('../input_data/chem_models/export_PYTHON_const_pressure.pln' ,grid, thermo, bc, U0, savingInterval=10, savingPath='../data', reactorModel='cst_p')
    #solver = db_gen.solver(grid, thermo, bc, U0, savingPath='data', reactorModel='cst_rho')
    # Total number of particles per unit volume #/m3

    # Masse d'une particule dans une section donnée (kg)
    m_class = grid.meanParticuleVolume * SootModel.RHO_SOOT
    ns = U0[3 + grid.nspecies:, :].T

    M0_prev = torch.sum(ns, dim=1)[0]
    M1_prev = torch.sum(ns * m_class, dim=1)[0]
    solver.solve(0.0005, 100, 0.95, 'sectional.json')
    M0 = torch.sum(grid.cells.ns, dim=1)[0]
    M1 = torch.sum(grid.cells.ns * m_class, dim=1)[0]


    print(f'number of particules before coagulation = {M0_prev:.5e} | number of particules after coagulation = {M0:.5e}')
    print(f'mass of particules before coagulation = {M1_prev:.5e} | mass of particules after coagulation = {M1:.5e}')


    post_proc = pp.PostProcessing('../simulation_results/test', 'sectional.json')
    #post_proc.plotSnapchot('T', 0)
    #post_proc.animateField('N_0')
    #post_proc.plotSpatialSootEvolution2D(-1)
    #post_proc.plotSpatialSootEvolution3D(-1, y_axis='log_size')
    #post_proc.plotTemporalSootEvolution2D(0)
    #post_proc.plotTemporalSootEvolution3D(0, y_axis='log_size')
    #post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
    for i in range(nclasses):
        post_proc.plotSnapchot(f'N_{i}', -1)
        post_proc.animateField(f'N_{i}')
    #post_proc.plotVariables(['T', 'P'], 0)
    #post_proc.animateField('P')
    #post_proc.plotVariable('P', 0)
    #post_proc.animateField('Y_C2H2')