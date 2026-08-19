import torch
import numpy as np
import cantera as ct
import multiprocessing as mp
from scipy.integrate import solve_ivp
from NeuralCFD.chemestry.DummyChemistry import DummyChemistry
from multiprocessing import shared_memory


_gas = None

def _init_worker(mech):
    global _gas
    _gas = ct.Solution(mech)


def _integrate_cell_cst_rho(args):
    i, T0, Y0, rho, dt = args
    gas = _gas
    nspecies = len(Y0)

    y0 = np.empty(nspecies + 1)
    y0[0] = T0
    y0[1:] = Y0

    # Pré-calcul des poids moléculaires (constants !)
    gas.TDY = T0, rho, Y0
    W = gas.molecular_weights.copy()  # constant, calculé une fois

    def rhs(t, y):
        T = y[0]
        Y = y[1:]
        gas.TDY = T, rho, Y

        omega = gas.net_production_rates
        ubar  = gas.partial_molar_int_energies
        cv    = gas.cv_mass

        oW    = omega * W
        dYdt  = oW / rho
        dTdt  = -np.dot(oW, ubar / W) / (rho * cv)

        dydt    = np.empty_like(y)
        dydt[0] = dTdt
        dydt[1:]= dYdt
        return dydt
    n = nspecies + 1

    sol = solve_ivp(
        rhs, (0.0, dt), y0,
        method='BDF',
        atol=1e-22,  # 1e-12
        rtol=0,  # 1e-8
        vectorized=False,
        dense_output=False,  # désactive l'interpolation inutile
    )

    if not sol.success:
        return i, T0, Y0, np.nan

    y_end = sol.y[:, -1]
    gas.TDY = y_end[0], rho, y_end[1:]
    return i, y_end[0], y_end[1:], gas.P


class canteraChemMultiproc(DummyChemistry):

    def __init__(self, mech, ncells, nproc=None):
        self.mech   = mech
        self.ncells = ncells
        self.nproc  = nproc or mp.cpu_count()
        self.pool   = mp.Pool(
            processes=self.nproc,
            initializer=_init_worker,
            initargs=(self.mech,)
        )

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def computeIntegratedTPY(self, grid, dt, model="cst_rho"):
        if model != "cst_rho":
            raise NotImplementedError

        T_np   = grid.cells.T.detach().cpu().numpy()
        Y_np   = grid.cells.Ys.detach().cpu().numpy()
        rho_np = grid.cells.rho.detach().cpu().numpy()

        tasks = [
            (i, T_np[i], Y_np[i], rho_np[i], dt)
            for i in range(self.ncells)
        ]

        # Chunksize : empiriquement ~32-64 est meilleur que ncells/(4*nproc)
        chunksize = max(1, min(64, self.ncells // self.nproc))
        results = self.pool.map(_integrate_cell_cst_rho, tasks, chunksize=chunksize)

        # Reconstruction vectorisée — pas de boucle Python
        results.sort(key=lambda x: x[0])  # garantir l'ordre
        T_arr = np.array([r[1] for r in results])
        Y_arr = np.array([r[2] for r in results])
        P_arr = np.array([r[3] for r in results])

        device = grid.cells.T.device
        dtype  = grid.cells.T.dtype

        return (
            torch.from_numpy(T_arr).to(device=device, dtype=dtype),
            torch.from_numpy(P_arr).to(device=device, dtype=dtype),
            torch.from_numpy(Y_arr).to(device=device, dtype=dtype),
        )