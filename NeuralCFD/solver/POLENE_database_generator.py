import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.transport.rusanov as rus
import NeuralCFD.transport.AUSMPlusUp as AUSMpup
import NeuralCFD.chemestry.canteraChemMultiproc as cantMP
import NeuralCFD.chemestry.canteraChem as cant
import NeuralCFD.chemestry.kineticsGPU as GPUchem
import torch
import time
import matplotlib.pyplot as plt
import json
import os
import h5py
import numpy as np
# TODO: create a ABC class for solver parent that defines general functions like saving

class solver:

    def __init__(self, grid, thermo, boundaryConditions, initialConditions, savingPath=None, reactorModel='cst_rho', printAdvance=True, savingInterval=500, dtypeDisk=np.float64):
        self.grid = grid
        self.thermo = thermo
        self.rusanovTransport = rus.rusanovFlux()
        self.chemistry = cantMP.canteraChemMultiproc(thermo.thermoFilePath, grid.ncells, 10)
        #self.chemistry = cant.canteraChem(thermo.thermoFilePath, grid.ncells)
        #self.chemistry = GPUchem.kineticsGPU(thermo.thermoFilePath, grid.ncells, grid.device)

        self.t = 0
        self.step = 0

        self.savingInterval = savingInterval

        self.boundaryConditions = boundaryConditions
        self.initialConditions = initialConditions
        self.reactorModel = reactorModel

        self.history_X = self.resetHistory()
        self.history_Y = self.resetHistory()

        self.savingPath = savingPath
        self.printAdvance = printAdvance

        self.dtypeDisk = dtypeDisk

    def resetHistory(self):
        hist = {'t': [], 'T': [], 'P': []}
        for spe in self.thermo.thermoProp.species:
            hist[f'Y_{spe}'] = []
        return(hist)

    def oneStep(self, dt, CFL):

        smax = torch.max(self.rusanovTransport.max_wave_speed(self.grid.cells)).item()
        dt_cfl_1 = torch.min(self.grid.dx / (smax + 1e-12) * CFL) # TODO: work precisely on cfl condition with variable cell size in all solvers
        # assert networkdt <= dt_cfl_1, 'networkdt must be less than dt computed with cfl=1 otherwise it will crash'
        N_sub_transport = int(dt / 2 / dt_cfl_1)

        subed_dt = 0.0
        for _ in range(N_sub_transport):
            smax = self.rusanovTransport.transport(dt_cfl_1, self.grid, self.boundaryConditions)
            subed_dt += dt_cfl_1
        smax = self.rusanovTransport.transport(dt / 2 - N_sub_transport * dt_cfl_1, self.grid, self.boundaryConditions)
        subed_dt += dt / 2 - N_sub_transport * dt_cfl_1

        self.save_X()
        self.chemistry.chemistry(dt, self.grid, self.thermo, self.boundaryConditions, self.reactorModel) #TODO: the chemestry should append at fixed rho to be coherent with strang spliting
        self.save_Y()

        subed_dt = 0.0
        for _ in range(N_sub_transport):
            smax = self.rusanovTransport.transport(dt_cfl_1, self.grid, self.boundaryConditions)
            subed_dt += dt_cfl_1
        smax = self.rusanovTransport.transport(dt / 2 - N_sub_transport * dt_cfl_1, self.grid, self.boundaryConditions)
        subed_dt += dt / 2 - N_sub_transport * dt_cfl_1

        self.t += dt
        self.step += 1

    def setIntialConditions(self):
        self.grid.updateCellsState(self.initialConditions, self.boundaryConditions, usePrevT=False) # delet initial conditions to gain disk space ?

    def save_X(self):
        self.history_X['t'].append(self.t)
        self.history_X['T'].append(self.grid.cells.T.detach().clone().cpu())
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history_X[f'Y_{spe}'].append(self.grid.cells.Ys[:, i].detach().clone().cpu())
        self.history_X['P'].append(self.grid.cells.p.detach().clone().cpu())

    def save_Y(self):
        self.history_Y['t'].append(self.t)
        self.history_Y['T'].append(self.grid.cells.T.detach().clone().cpu())
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history_Y[f'Y_{spe}'].append(self.grid.cells.Ys[:, i].detach().clone().cpu())
        self.history_Y['P'].append(self.grid.cells.p.detach().clone().cpu())


    def writeResults(self, caseName):

        # TODO: explorer l'utilisation de float 16 avec normalisation log des espèces

        if self.savingPath is None:
            raise ValueError('No saving path specified')

        if caseName is None:
            caseName = "results_default_sim.h5"

        filepath = os.path.join(self.savingPath, caseName)

        if not os.path.exists(self.savingPath):
            os.makedirs(self.savingPath)

        if len(self.history_X['t']) == 0:
            return

        with h5py.File(filepath, "a") as f:

            if "x" not in f:
                f.create_dataset(
                    "x",
                    data=self.grid.x.detach().cpu().numpy().astype(self.dtypeDisk),
                    compression="lzf"
                )

            def stack_and_write(group, history):

                t = np.array(history['t'], dtype=self.dtypeDisk)

                T = np.stack([t.cpu().numpy() for t in history['T']]).astype(self.dtypeDisk)
                P = np.stack([p.cpu().numpy() for p in history['P']]).astype(self.dtypeDisk)

                def append(name, data):

                    if name not in f:
                        maxshape = (None,) + data.shape[1:]
                        f.create_dataset(
                            name,
                            data=data,
                            maxshape=maxshape,
                            chunks=True,
                            compression="lzf"
                        )
                    else:
                        dset = f[name]
                        old = dset.shape[0]
                        new = old + data.shape[0]
                        dset.resize(new, axis=0)
                        dset[old:new] = data

                append(f"{group}/t", t)
                append(f"{group}/T", T)
                append(f"{group}/P", P)

                for spe in self.thermo.thermoProp.species:
                    Y = np.stack([
                        y.cpu().numpy() for y in history[f"Y_{spe}"]
                    ]).astype(self.dtypeDisk)

                    append(f"{group}/Y_{spe}", Y)

            stack_and_write("X", self.history_X)
            stack_and_write("Y", self.history_Y)

        self.history_X = self.resetHistory()
        self.history_Y = self.resetHistory()

    def solve(self, tmax, maxiter, dtMax, CFL, caseName=None):
        t0 = time.time()

        self.setIntialConditions()

        # load self.* that are used in the loop to be a bit faster
        while self.t < tmax and self.step < maxiter:

            self.oneStep(dtMax, CFL)

            if self.printAdvance:
                print(f'time: {self.t} | dt : {dtMax} | step: {self.step}')

            if self.step % self.savingInterval == 0:
                self.writeResults(caseName)

        if self.printAdvance:
            print('simulation took %.10f seconds' % (time.time() - t0))
            print(f"Finished in {self.step} steps, final time approx {self.history_Y['t'][-1]:.6e} s")

        self.writeResults(caseName)


if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'C:\Users\bromain\anaconda3\Lib\site-packages\cantera\data\gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)

    solver = solver(grid, thermo)

    solver.solve(0.005, 1000, 0.9)