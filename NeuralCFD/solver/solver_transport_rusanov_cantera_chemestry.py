import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.transport.rusanov as rus
import NeuralCFD.chemestry.canteraChem as cant
import torch
import time
import json
import os
# TODO: create a ABC class for solver parent that defines general functions like saving

class solver:

    def __init__(self, grid, thermo, boundaryConditions, initialConditions, savingInterval, savingPath=None, reactorModel='cst_rho'):
        self.grid = grid
        self.thermo = thermo
        self.rusanovTransport = rus.rusanovFlux()
        self.chemistry = cant.canteraChem(thermo.thermoFilePath, grid.ncells)

        self.t = 0
        self.step = 0

        self.total_transport_time = 0
        self.total_chemistry_time = 0

        self.savingInterval = savingInterval

        self.boundaryConditions = boundaryConditions
        self.initialConditions = initialConditions
        self.reactorModel = reactorModel

        self.history = {'t': [], 'T': [], 'rho': [], 'P': [], 'u':[]}
        for spe in self.thermo.thermoProp.species:
            self.history[f'Y_{spe}'] = []

        self.savingPath = savingPath

    def oneStep(self, dt):
        ti = time.perf_counter()
        smax = self.rusanovTransport.transport(0.5 * dt, self.grid, self.boundaryConditions)
        self.total_transport_time += time.perf_counter()-ti

        ti = time.perf_counter()
        self.chemistry.chemistry(dt, self.grid, self.thermo, self.boundaryConditions, self.reactorModel) #TODO: the chemestry should append at fixed rho to be coherent with strang spliting
        self.total_chemistry_time += time.perf_counter()-ti

        ti = time.perf_counter()
        smax = self.rusanovTransport.transport(0.5 * dt, self.grid, self.boundaryConditions)
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

    def solve(self, tmax, maxiter, dtMax, caseName=None):
        t0 = time.time()

        self.setIntialConditions()

        # load self.* that are used in the loop to be a bit faster
        while self.t < tmax and self.step < maxiter:
            smax = torch.max(self.rusanovTransport.max_wave_speed(self.grid.cells)).item()
            dt_cfl_1 = self.grid.dx / (smax + 1e-12)
            #assert networkdt <= dt_cfl_1, 'networkdt must be less than dt computed with cfl=1 otherwise it will crash'
            dt = min(dt_cfl_1, dtMax)
            self.oneStep(dt)

            print(f'time: {self.t} | dt : {dt} | step: {self.step}')

            if self.step % self.savingInterval == 0 or self.t >= tmax:
                self.save()

        print('simulation took %.10f seconds' % (time.time() - t0))
        print(f"Finished in {self.step} steps, final time approx {self.history['t'][-1]:.6e} s")
        print('======== Perf ========')
        print('Total transport time: {:.5f}'.format(self.total_transport_time))
        print('Mean transport time per step: {:.5f}'.format(self.total_transport_time / self.step))

        print('Total chemistry time: {:.5f}'.format(self.total_chemistry_time))
        print('Mean chemistry time per step: {:.5f}'.format(self.total_chemistry_time / self.step))

        self.writeResults(caseName)


if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'C:\Users\bromain\anaconda3\Lib\site-packages\cantera\data\gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)

    solver = solver(grid, thermo)

    solver.solve(0.005, 1000, 0.9)