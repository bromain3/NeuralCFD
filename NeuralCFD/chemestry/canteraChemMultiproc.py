import torch
import numpy as np
import cantera as ct
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from scipy.integrate import solve_ivp
from NeuralCFD.chemestry.DummyChemistry import DummyChemistry

# ============================================================
# Worker global (1 ct.Solution par process)
# ============================================================

_gas = None

def _init_worker(mech):
    global _gas
    _gas = ct.Solution(mech)


def _integrate_cell_cst_rho(args):
    """
    Intégration chimique d'une cellule à densité constante.
    """
    i, T0, Y0, rho, dt = args
    gas = _gas
    nspecies = len(Y0)

    y0 = np.empty(nspecies + 1)
    y0[0] = T0
    y0[1:] = Y0

    def rhs(t, y):
        T = y[0]
        Y = y[1:]

        gas.TDY = T, rho, Y

        omega = gas.net_production_rates              # kmol / m3 / s
        W = gas.molecular_weights                     # kg / kmol
        ubar = gas.partial_molar_int_energies         # J / kmol
        cv = gas.cv_mass                              # J / kg / K

        dYdt = (omega * W) / rho                      # 1 / s
        u_k = ubar / W                                # J / kg
        dTdt = - np.dot(omega * W, u_k) / (rho * cv)  # K / s

        dydt = np.empty_like(y)
        dydt[0] = dTdt
        dydt[1:] = dYdt
        return dydt

    sol = solve_ivp(
        rhs,
        (0.0, dt),
        y0,
        method='BDF',
        # atol=1e-12,
        # rtol=1e-8,
        vectorized=False,
        # jac=jacobian,
    )

    if not sol.success:
        raise ValueError('AAAAAAAAAAAAAAAAAAAAAAAA')
        return i, T0, Y0, np.nan

    y_end = sol.y[:, -1]

    gas.TDY = y_end[0], rho, y_end[1:]
    P = gas.P

    return i, y_end[0], y_end[1:], P


# ============================================================
# Classe principale
# ============================================================

class canteraChemMultiproc(DummyChemistry):

    def __init__(self, mech, ncells, nproc=None):
        self.mech = mech
        self.ncells = ncells
        self.nproc = nproc or mp.cpu_count()

        self.pool = mp.Pool(
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
            raise NotImplementedError(
                'Only model="cst_rho" is supported (physically consistent)'
            )

        # --- Extraction NumPy ---
        T_np = grid.cells.T.detach().cpu().numpy()
        Y_np = grid.cells.Ys.detach().cpu().numpy()
        rho_np = grid.cells.rho.detach().cpu().numpy()

        tasks = [
            (i, T_np[i], Y_np[i], rho_np[i], dt)
            for i in range(self.ncells)
        ]

        # --- Parallélisation ---
        results = self.pool.map(
            _integrate_cell_cst_rho,
            tasks,
            chunksize=max(1, self.ncells // (4 * self.nproc))
        )

        # --- Reconstruction torch ---
        device = grid.cells.T.device
        dtype = grid.cells.T.dtype

        T_new = torch.empty_like(grid.cells.T)
        P_new = torch.empty_like(grid.cells.T)
        Ys_new = torch.empty_like(grid.cells.Ys)

        for i, T, Y, P in results:
            T_new[i] = T
            Ys_new[i] = torch.from_numpy(Y)
            P_new[i] = P

        return (
            T_new.to(device=device, dtype=dtype),
            P_new.to(device=device, dtype=dtype),
            Ys_new.to(device=device, dtype=dtype),
        )
