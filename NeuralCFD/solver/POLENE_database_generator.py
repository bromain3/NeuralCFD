import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.transport.rusanov as rus
import NeuralCFD.chemestry.canteraChem as cant
import torch
import time
import matplotlib.pyplot as plt
import json
import os
# TODO: create a ABC class for solver parent that defines general functions like saving

class solver:

    def __init__(self, grid, thermo, boundaryConditions, initialConditions, savingPath=None, reactorModel='cst_rho', printAdvance=True, savingInterval=500):
        self.grid = grid
        self.thermo = thermo
        self.rusanovTransport = rus.rusanovFlux()
        self.chemistry = cant.canteraChem(thermo.thermoFilePath, grid.ncells)

        self.t = 0
        self.step = 0

        self.savingInterval = savingInterval

        self.boundaryConditions = boundaryConditions
        self.initialConditions = initialConditions
        self.reactorModel = reactorModel

        self.history_X = {'t': [], 'T': [], 'P': []}
        for spe in self.thermo.thermoProp.species:
            self.history_X[f'Y_{spe}'] = []

        self.history_Y = {'t': [], 'T': [], 'P': []}
        for spe in self.thermo.thermoProp.species:
            self.history_Y[f'Y_{spe}'] = []

        self.savingPath = savingPath
        self.printAdvance = printAdvance

    def oneStep(self, dt, CFL):

        smax = torch.max(self.rusanovTransport.max_wave_speed(self.grid.cells)).item()
        dt_cfl_1 = self.grid.dx / (smax + 1e-12) * CFL
        # assert networkdt <= dt_cfl_1, 'networkdt must be less than dt computed with cfl=1 otherwise it will crash'
        N_sub_transport = int(dt / 2 / dt_cfl_1)

        subed_dt = 0.0
        for _ in range(N_sub_transport):
            smax = self.rusanovTransport.transport(dt_cfl_1, self.grid, self.boundaryConditions)
            subed_dt += dt_cfl_1
        smax = self.rusanovTransport.transport(dt / 2 - N_sub_transport * dt_cfl_1, self.grid, self.boundaryConditions)
        subed_dt += dt / 2 - N_sub_transport * dt_cfl_1
        print(subed_dt, dt)

        self.save_X()
        self.chemistry.chemistry(dt, self.grid, self.thermo, self.boundaryConditions, self.reactorModel) #TODO: the chemestry should append at fixed rho to be coherent with strang spliting
        self.save_Y()

        subed_dt = 0.0
        for _ in range(N_sub_transport):
            smax = self.rusanovTransport.transport(dt_cfl_1, self.grid, self.boundaryConditions)
            subed_dt += dt_cfl_1
        smax = self.rusanovTransport.transport(dt / 2 - N_sub_transport * dt_cfl_1, self.grid, self.boundaryConditions)
        subed_dt += dt / 2 - N_sub_transport * dt_cfl_1
        print(subed_dt, dt)

        self.t += dt
        self.step += 1

    def setIntialConditions(self):
        self.grid.updateCellsState(self.initialConditions, self.boundaryConditions, usePrevT=False) # delet initial conditions to gain disk space ?

    def save_X(self):
        self.history_X['t'].append(self.t)
        self.history_X['T'].append(list(self.grid.cells.T.detach().clone().cpu().numpy()))
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history_X[f'Y_{spe}'].append(list(self.grid.cells.Ys[:, i].detach().clone().cpu().numpy()))
        self.history_X['P'].append(list(self.grid.cells.p.detach().clone().cpu().numpy()))

    def save_Y(self):
        self.history_Y['t'].append(self.t)
        self.history_Y['T'].append(list(self.grid.cells.T.detach().clone().cpu().numpy()))
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history_Y[f'Y_{spe}'].append(list(self.grid.cells.Ys[:, i].detach().clone().cpu().numpy()))
        self.history_Y['P'].append(list(self.grid.cells.p.detach().clone().cpu().numpy()))

    def addStaticSave(self):
        self.history_X['x'] = list(self.grid.x.detach().clone().cpu().numpy())
        self.history_Y['x'] = list(self.grid.x.detach().clone().cpu().numpy())

    def writeResults(self, caseName):
        if self.savingPath is not None:
            self.addStaticSave()
            if caseName is None:
                caseName = 'results_default_sim.json'
            if not os.path.exists(self.savingPath):
                os.makedirs(self.savingPath)
            with open(os.path.join(self.savingPath, os.path.splitext(caseName)[0] + '_X' + os.path.splitext(caseName)[1]), 'w') as f:
                json.dump(self.history_X, f)
            with open(os.path.join(self.savingPath, os.path.splitext(caseName)[0] + '_Y' + os.path.splitext(caseName)[1]), 'w') as f:
                json.dump(self.history_Y, f)

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