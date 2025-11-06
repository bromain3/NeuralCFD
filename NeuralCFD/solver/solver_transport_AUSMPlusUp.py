import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.transport.AUSMPlusUp as AUSMPlusUp
import NeuralCFD.utilities_functions.primitive_converter as pc

import torch
import time
import matplotlib.pyplot as plt
import json
import os
import gc

# TODO: create a ABC class for solver parent that defines general functions like saving
class solver:

    def __init__(self, grid, thermo, boundaryConditions, initialConditions, savingInterval, savingPath=None):
        self.grid = grid
        self.thermo = thermo
        self.AUSMFlux = AUSMPlusUp.AUSMPlusUpFlux()

        self.t = 0
        self.step = 0

        self.savingInterval = savingInterval

        self.total_transport_time = 0

        self.boundaryConditions = boundaryConditions
        self.initialConditions = initialConditions

        self.history = {'t': [], 'T': [], 'rho': [], 'P': [], 'u':[]}
        for spe in self.thermo.thermoProp.species:
            self.history[f'Y_{spe}'] = []

        self.savingPath = savingPath

    def oneStep(self, dt):
        ti = time.perf_counter()
        smax = self.AUSMFlux.transport(dt, self.grid, self.boundaryConditions)
        self.total_transport_time += time.perf_counter()-ti
        self.t += dt
        self.step += 1

    def setIntialConditions(self):
        self.grid.updateCellsState(self.initialConditions, self.boundaryConditions, usePrevT=False) # delet initial conditions to gain disk space ?

    def save(self):
        self.history['t'].append(self.t)
        self.history['T'].append(list(self.grid.cells.T.detach().clone().cpu().numpy()))
        self.history['rho'].append(list(self.grid.cells.rho.detach().clone().cpu().numpy()))
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history[f'Y_{spe}'].append(list(self.grid.cells.Ys[:, i].detach().clone().cpu().numpy()))
        self.history['P'].append(list(self.grid.cells.p.detach().clone().cpu().numpy()))
        self.history['u'].append(list(self.grid.cells.u.detach().clone().cpu().numpy()))

    def addStaticSave(self):
        self.history['x'] = list(self.grid.x.detach().clone().cpu().numpy())

    def writeResults(self, caseName):
        if self.savingPath is not None:
            self.addStaticSave()
            if caseName is None:
                caseName = 'results_default_sim.json'
            if not os.path.exists(self.savingPath):
                os.makedirs(self.savingPath)
            with open(os.path.join(self.savingPath, caseName), 'w') as f:
                json.dump(self.history, f)

    def solve(self, tmax, maxiter, CFL, caseName=None):
        t0 = time.time()

        self.setIntialConditions()

        torch.cuda.empty_cache() # removes from the cache heavy temperature table made for initialisation
        gc.collect() # removes from the cache heavy temperature table made for initialisation

        print(f"Max mémoire used : {torch.cuda.memory.max_memory_allocated() / 1024 ** 2:.2f} Mb")
        print(f"Max mémoire reserved : {torch.cuda.memory.max_memory_reserved() / 1024 ** 2:.2f} Mb")

        # load self.* that are used in the loop to be a bit faster
        while self.t < tmax and self.step < maxiter:
            smax = torch.max(self.AUSMFlux.max_wave_speed(self.grid.cells)).item()
            dt_cfl = CFL * self.grid.dx / (smax + 1e-12)
            dt = min(dt_cfl, tmax - self.t)

            self.oneStep(dt)

            print(f'time: {self.t} | step: {self.step}')

            if self.step % self.savingInterval == 0 or self.t >= tmax:
                self.save()

        print('simulation took %.10f seconds' % (time.time() - t0))
        print(f"Finished in {self.step} steps, final time approx {self.history['t'][-1]:.6e} s")
        print('======== Perf ========')
        print('Total transport time: {:.5f}'.format(self.total_transport_time))
        print('Mean transport time per step: {:.5f}'.format(self.total_transport_time / self.step))

        self.writeResults(caseName)


if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'C:\Users\bromain\anaconda3\Lib\site-packages\cantera\data\gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)

    solver = solver(grid, thermo)

    solver.solve(0.005, 1000, 0.9)