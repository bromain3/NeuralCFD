import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.soot_chemistry.growth.LeungLindset import LeungLindset as growthLL
from NeuralCFD.soot_chemistry.growth.Lindset import Lindset as growthLIN
from NeuralCFD.soot_chemistry.nucleation.LeungLindset import LeungLindset as nucLL
from NeuralCFD.soot_chemistry.nucleation.Lindset import Lindset as nucLIN
from NeuralCFD.soot_chemistry.oxidation.LeungLindset import LeungLindset as oxiLL
from NeuralCFD.soot_chemistry.oxidation.HACA import HACA as oxiHACA
from NeuralCFD.soot_chemistry.growth.HACA import HACA as growthHACA
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular import FreeMolecular
from NeuralCFD.soot_chemistry.coagulation.FreeMolecular_vect import FreeMolecular_vect
from NeuralCFD.soot_chemistry.coagulation.Fuchs_vect import Fuchs_vect
from NeuralCFD.soot_chemistry.SootModel import SootModel
from NeuralCFD.chemestry.canteraChem import canteraChem
import NeuralCFD.solver.solver_transport_chemistry_soot_subcycle as ssub
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
from NeuralCFD.utilities_functions.torchInterp1D import torch_interp
import cantera as ct

import torch
import numpy as np
import random


# TODO: add functions rho from and P from TP so that perfect gas assumption is centralized if we want to change

def phi2Y(phi):
    MMCH4 = 16.04
    MMO2 = 31.999
    O2_N2 = 3.32

    mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
    Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
    return (Y)


if __name__ == "__main__":

    ######################################
    #### General variables definition ####
    ######################################
    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    ncells = 1
    grid_length = 1.0
    nclasses = 40

    th_file = r'../input_data/gri30.yaml'

    soot_d_min = 1.145853e-9 ; soot_d_max = 10_000.0e-9 # size used for LeungLindset model to ensure a nucleation of a particle at exactly 100 particles of carbon (if rho soot = 1800) as stated in the paper
    #soot_d_min = 0.526715e-9 ; soot_d_max = 10_000.0e-9 # size used for Lindset model to ensure a nucleation of a particle at exactly 10 particles of carbon (if rho soot = 1800) as stated in the paper (Naphtalene)



    #############################################################
    ########## check if grid is smaller than particles ##########
    #############################################################
    print()
    print(f'Biggest particle diameter: {soot_d_max:.5e} m')
    print(f'cell size: {grid_length / ncells:.5e} m')
    print()

    #####################################
    ########## object creation ##########
    #####################################
    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    print()
    print(f'Thermo created with file: {th_file}')
    print()

    grid = NeuralCFD.grid.grid.grid(grid_length, ncells, thermo, device=device, dtype=dtype,
                                    sectionalParams={"dmin": soot_d_min, "dmax": soot_d_max, "nclasses": nclasses,
                                                     'spacing': 'log'})
    chem = canteraChem(mech=th_file, ncells=ncells)
    print()
    if nclasses > 0:
        print(
            f'Grid created with sectional model enabled. Sectional model params are d_min={soot_d_min} and d_max={soot_d_max} with nclasses={nclasses}')
    else:
        print(f'Grid created with sectional model disabled.')
    print()

    ######################################
    ###### flamme speed computation ######
    ######################################
    import matplotlib.pyplot as plt

    ####################################
    ######## creating the field ########
    ####################################
    gas = ct.Solution(thermo.thermoFilePath)

    gas.TPY = 1700, 101325, {'CH4': 0.05483547715890419,'O2':0.21878808399099442, 'N2': 0.7263764388501015}
    gas.set_equivalence_ratio(2.34, fuel='CH4', oxidizer='O2:1.0, N2:3.76')

    p0 = 101325 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 =  1700 + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
    ns0 = torch.zeros((grid.ncells, grid.nclasses), device=device, dtype=dtype)
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, ns0, thermo)

    #####################################
    ######## boundary conditions ########
    #####################################
    bc = {'type': 'closed'}

    #######################################
    ######## solvers Configuration ########
    #######################################

    # sootSolver = FrenklachetWang_Leung_nucleation(grid, thermo)
    sootSolver = SootModel(nucLL, oxiLL, growthLL, Fuchs_vect, grid, thermo)

    solver = ssub.solver(grid, thermo, rusanovFlux(), chem, sootSolver, bc, U0, savingInterval=1, simdt=1e-7 , savingPath='../simulation_results/0D_soot', initialGridTemperature=T0)

    #solver.solve(3.5e-4, 30_000, 0.9, f'data.json')


    post_proc = pp.PostProcessing('../simulation_results/0D_soot',f'data.json')

    if nclasses > 0:
        #post_proc.animateField('T')
        #post_proc.plotVariables(['T'], 0)
        #post_proc.plotVariables(['N_15'], 0)
        post_proc.plotTemporalSootEvolution3D(0, log_scale=True, y_axis='log_size')
        # post_proc.plotSpatialSootEvolution2D(-1)
        #post_proc.plotSpatialSootEvolutionAnimated3D(y_axis='log_size')
        #post_proc.plotSootVolumeFraction(-1, log_scale=True, unit='ppm', expData=exp_data[equivalence_ratio])
        #post_proc.animateSootVolumeFraction()
    else:
        post_proc.plotSnapchot('T', -1)