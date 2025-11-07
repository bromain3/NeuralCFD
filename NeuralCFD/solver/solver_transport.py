import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.solver.DummySolver import DummySolver
import torch
import time
import json
import os
import gc

class solver(DummySolver):

    def __init__(self, grid, thermo, transportSolver, boundaryConditions, initialConditions, savingInterval, savingPath=None):
        DummySolver.__init__(self, grid, thermo, boundaryConditions, initialConditions, savingInterval, savingPath=savingPath)

        self.transport = transportSolver

        self.total_transport_time = 0


    def oneStep(self, dt):
        ti = time.perf_counter()
        smax = self.transport.transport(dt, self.grid, self.boundaryConditions)
        self.total_transport_time += time.perf_counter()-ti
        self.t += dt
        self.step += 1


    def solve(self, tmax, maxiter, CFL, caseName=None):
        t0 = time.time()

        self.setIntialConditions()

        torch.cuda.empty_cache() # removes from the cache heavy temperature table made for initialisation
        gc.collect() # removes from the cache heavy temperature table made for initialisation

        print(f"Max mémoire used : {torch.cuda.memory.max_memory_allocated() / 1024 ** 2:.2f} Mb")
        print(f"Max mémoire reserved : {torch.cuda.memory.max_memory_reserved() / 1024 ** 2:.2f} Mb")

        # load self.* that are used in the loop to be a bit faster
        while self.t < tmax and self.step < maxiter:
            smax = torch.max(self.transport.max_wave_speed(self.grid.cells)).item()
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