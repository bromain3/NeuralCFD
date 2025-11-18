import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
from NeuralCFD.solver.DummySolver import DummySolver
import torch
import time
import gc


class solver(DummySolver):

    def __init__(self, grid, thermo, transportSolver, chemistrySolver, boundaryConditions, initialConditions, savingInterval, savingPath=None, reactorModel='cst_rho', initialGridTemperature=False):
        DummySolver.__init__(self, grid, thermo, boundaryConditions, initialConditions, savingInterval, savingPath=savingPath, initialGridTemperature=initialGridTemperature)

        self.Transport = transportSolver
        self.chemistry = chemistrySolver

        self.total_transport_time = 0
        self.total_chemistry_time = 0

        self.reactorModel = reactorModel


    def oneStep(self, dt):
        ti = time.perf_counter()
        smax = self.Transport.transport(0.5 * dt, self.grid, self.boundaryConditions)
        self.total_transport_time += time.perf_counter()-ti

        ti = time.perf_counter()
        self.chemistry.chemistry(dt, self.grid, self.thermo, self.boundaryConditions, self.reactorModel) #TODO: the chemestry should append at fixed rho to be coherent with strang spliting
        self.total_chemistry_time += time.perf_counter()-ti

        ti = time.perf_counter()
        smax = self.Transport.transport(0.5 * dt, self.grid, self.boundaryConditions)
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
            smax = torch.max(self.Transport.max_wave_speed(self.grid.cells)).item()
            dt_cfl = torch.min(self.grid.dx).item() / (smax + 1e-12) * CFL # TODO: This dt_cfl is overestimated. it should be recomputed internaly as dt_loc = dxi/si so we should take min(dt_loc) and not min(num) / max(denom)
            #assert networkdt <= dt_cfl_1, 'networkdt must be less than dt computed with cfl=1 otherwise it will crash'
            #dt = min(dt_cfl_1, dtMax)
            dt = min(dt_cfl, tmax - self.t)
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