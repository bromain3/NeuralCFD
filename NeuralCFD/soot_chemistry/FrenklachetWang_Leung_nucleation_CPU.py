"""
Frenklach et Wang with Leung nucleation - CPU version with scipy integration and multiprocessing.

Features:
- Nucleation integrated in BDF system
- Same HACA surface growth with active site fraction
- Same oxidation (O2 + OH)
- Same Brownian coagulation kernel with Cunningham correction
- CPU multiprocessing parallelization (like cantera chemistry)

Author: CPU-optimized parallel version
"""

import torch
import numpy as np
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from scipy.integrate import solve_ivp
from NeuralCFD.soot_chemistry.DummySoot import DummySoot


# Physical constants
R = 8.314  # J/(mol·K)
kB = 1.380649e-23  # J/K
RHO_SOOT = 1800.0  # kg/m³
lambda_g = 68e-9  # Mean free path (m)
mu = 1.8e-5  # Dynamic viscosity (Pa·s)


# ============================================================
# Worker global variables (one set per process)
# ============================================================

_worker_data = None


def _init_worker(radii, surfaces, cc, coag_map, sectionSize, nclasses,
                 chi_s, mC, A1, E1, A2, n2, E2, A3, E3):
    """Initialize worker process with pre-computed data."""
    global _worker_data
    _worker_data = {
        'radii': radii,
        'surfaces': surfaces,
        'cc': cc,
        'coag_map': coag_map,
        'sectionSize': sectionSize,
        'nclasses': nclasses,
        'chi_s': chi_s,
        'mC': mC,
        'A1': A1, 'E1': E1,
        'A2': A2, 'n2': n2, 'E2': E2,
        'A3': A3, 'E3': E3,
    }


def _compute_active_site_fraction(T, C_H, C_C2H2):
    """Compute HACA active site fraction θ*."""
    data = _worker_data

    # Rate constants
    k1 = data['A1'] * np.exp(-data['E1'] / (R * T))
    k2 = data['A2'] * T ** data['n2'] * np.exp(-data['E2'] / (R * T))
    k3 = data['A3'] * np.exp(-data['E3'] / (R * T))

    # Active site fraction
    numerator = k1 * C_H
    denom = numerator + k2 * C_C2H2 + k3 * C_H + 1e-30

    theta_star = numerator / denom

    return theta_star, k2


def _compute_R_HACA(theta_star, k2, C_C2H2):
    """Compute HACA surface growth rate (kg/(m²·s))."""
    data = _worker_data
    mass_rate = k2 * C_C2H2 * theta_star * data['chi_s'] * (2 * data['mC'])
    return mass_rate


def _compute_oxidation_rate(theta_star, T, C_O2, C_OH):
    """Compute oxidation rate (kg/(m²·s))."""
    data = _worker_data

    # O2 oxidation
    A_O2 = 2.2e12 * 1e-6  # m³/(mol·s)
    E_O2 = 7_500.0  # J/mol
    k_O2 = A_O2 * np.exp(-E_O2 / (R * T))
    R_O2 = k_O2 * C_O2 * theta_star * data['chi_s'] * (2 * data['mC'])

    # OH oxidation
    gamma_OH = 0.13
    m_OH = 17e-3 / 6.022e23  # kg/molecule
    thermal_vel = np.sqrt(8 * kB * T / (np.pi * m_OH))
    R_OH = gamma_OH * C_OH * thermal_vel * theta_star * data['mC'] * data['chi_s']

    return R_O2 + R_OH


def _compute_brownian_kernel_matrix(T, mu_gas):
    """Compute Brownian coagulation kernel matrix."""
    data = _worker_data
    radii = data['radii']
    cc = data['cc']

    # Broadcasting
    ri = radii[:, None]
    rj = radii[None, :]
    cc_i = cc[:, None]
    cc_j = cc[None, :]

    # Prefactor
    prefactor = 2 * kB * T / (3 * mu_gas)

    # Kernel
    K = prefactor * (ri + rj) * (cc_i / ri + cc_j / rj)

    return K


def _nucleation_rate(T, C_C2H2, v_nuc):
    """
    Compute nucleation rate for first section.

    Returns:
        R_nuc_rate: Rate of particle formation (1/(m³·s))
    """
    k1 = 0.1e5 * np.exp(-21_000 / T)
    R_nuc = k1 * C_C2H2
    m_p = RHO_SOOT * v_nuc

    return R_nuc / m_p


def _surface_growth(ns, G, sectionSize):
    """Surface growth - flux in size space."""
    flux = G * ns

    dndt = np.zeros_like(ns)
    dndt[1:] += flux[:-1] / sectionSize[1:]
    dndt[:-1] -= flux[:-1] / sectionSize[:-1]

    return dndt


def _oxidation(ns, L, sectionSize):
    """Oxidation - reverse flux in size space."""
    flux = L * ns

    dndt = np.zeros_like(ns)
    dndt[:-1] += flux[1:] / sectionSize[:-1]
    dndt[1:] -= flux[1:] / sectionSize[1:]
    dndt[0] -= flux[0] / sectionSize[0]

    return dndt


def _coagulation(ns, K_cell, coag_map, nclasses):
    """Brownian coagulation optimized with symmetry."""
    dNdt = np.zeros_like(ns)

    for i in range(nclasses):
        ni = ns[i]
        if ni < 1e-30:
            continue

        for j in range(i, nclasses):
            nj = ns[j]
            if nj < 1e-30:
                continue

            beta = K_cell[i, j]
            rate = (0.5 if i == j else 1.0) * beta * ni * nj

            dNdt[i] -= rate
            if i != j:
                dNdt[j] -= rate

            dNdt[coag_map[i, j]] += rate

    return dNdt


def _integrate_cell_soot(args):
    """
    Integration of soot dynamics for one cell.

    Args:
        args: Tuple containing (cell_idx, ns0, T, rho, Ys, mu_gas, dt, indices, v_nuc)

    Returns:
        Tuple (cell_idx, ns_final)
    """
    cell_idx, ns0, T, rho, Ys, mu_gas, dt, indices, v_nuc = args
    data = _worker_data

    # Molecular weights
    MW_H = 1e-3  # kg/mol
    MW_C2H2 = 26e-3  # kg/mol
    MW_O2 = 32e-3  # kg/mol
    MW_OH = 17e-3  # kg/mol

    # Species concentrations (mol/m³)
    C_H = rho * Ys[indices['H']] / MW_H
    C_C2H2 = rho * Ys[indices['C2H2']] / MW_C2H2
    C_O2 = rho * Ys[indices['O2']] / MW_O2
    C_OH = rho * Ys[indices['OH']] / MW_OH

    # Compute rate parameters (constant during dt)
    theta_star, k2 = _compute_active_site_fraction(T, C_H, C_C2H2)
    R_HACA = _compute_R_HACA(theta_star, k2, C_C2H2)
    oxyd_rate = _compute_oxidation_rate(theta_star, T, C_O2, C_OH)
    K_cell = _compute_brownian_kernel_matrix(T, mu_gas)

    # Nucleation rate
    R_nuc_rate = _nucleation_rate(T, C_C2H2, v_nuc)

    # Growth and oxidation rates (kg/s for each size class)
    G_cell = data['surfaces'] * R_HACA
    L_cell = data['surfaces'] * oxyd_rate

    # RHS function including nucleation
    def rhs(t, ns):
        ns = np.maximum(ns, 0.0)

        # All processes
        dn_growth = _surface_growth(ns, G_cell, data['sectionSize'])
        dn_oxyd = _oxidation(ns, L_cell, data['sectionSize'])
        dn_coag = _coagulation(ns, K_cell, data['coag_map'], data['nclasses'])

        # Nucleation - only affects first section
        dn_nuc = np.zeros_like(ns)
        dn_nuc[0] = R_nuc_rate

        return dn_growth + dn_oxyd + dn_coag + dn_nuc

    # Integrate with scipy BDF
    sol = solve_ivp(
        rhs,
        [0, dt],
        ns0,
        method='RK45',
        vectorized=False,
    )

    ns_final = sol.y[:, -1]

    ns_final = np.maximum(ns_final, 0.0)

    return cell_idx, ns_final


# ============================================================
# Main class
# ============================================================

class FrenklachetWang_Leung_nucleation_CPU(DummySoot):
    """
    CPU-optimized Frenklach model with Leung nucleation and multiprocessing.
    Nucleation is now integrated inside the BDF solver.
    """

    def __init__(self, grid, thermo, C2H2name='C2H2', O2name='O2', OHname='OH',
                 Oname='O', Hname='H', nproc=None):
        """
        Initialize CPU-optimized parallel Frenklach model.

        Args:
            grid: Grid object containing particle size distribution
            thermo: Thermodynamic properties object
            C2H2name, O2name, OHname, Oname, Hname: Species names
            nproc: Number of processes (default: all CPUs)
        """
        self.device = grid.device

        # Species indices
        self.indices = {
            'C2H2': thermo.thermoProp.species.index(C2H2name),
            'OH': thermo.thermoProp.species.index(OHname),
            'O2': thermo.thermoProp.species.index(O2name),
            'O': thermo.thermoProp.species.index(Oname),
            'H': thermo.thermoProp.species.index(Hname),
        }

        # HACA kinetic constants (converted to m³)
        self.A1 = 4.2e13 * 1e-6  # m³/(mol·s)
        self.E1 = 13_000.0  # J/mol
        self.A2 = 8.0e7 * 1e-6  # m³/(mol·s)
        self.n2 = 1.56
        self.E2 = 3_800.0  # J/mol
        self.A3 = 1.0e13 * 1e-6  # m³/(mol·s)
        self.E3 = 0.0  # J/mol

        # Surface site density
        self.chi_s = 2.3e19  # sites/m²
        self.mC = 12e-3 / 6.022e23  # kg/molecule

        # Coagulation map
        self.coag_map = self._build_conservative_coag_map(grid)

        # Pre-compute geometry
        self._precompute_geometry(grid)

        # Store grid info
        self.nclasses = grid.nclasses
        self.sectionSize = grid.sectionSize.cpu().numpy()
        self.v_nuc = grid.meanParticuleVolume[0].cpu().item()
        self.ncells = grid.cells.ns.shape[0]

        # Multiprocessing setup
        self.nproc = nproc or mp.cpu_count()
        self.pool = mp.Pool(
            processes=self.nproc,
            initializer=_init_worker,
            initargs=(
                self.radii, self.surfaces, self.cc, self.coag_map,
                self.sectionSize, self.nclasses, self.chi_s, self.mC,
                self.A1, self.E1, self.A2, self.n2, self.E2, self.A3, self.E3
            )
        )

    def close(self):
        """Close the multiprocessing pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        """Destructor to ensure pool is closed."""
        self.close()

    def _build_conservative_coag_map(self, grid):
        """Vectorized coagulation map construction."""
        nclasses = grid.nclasses
        v = grid.meanParticuleVolume

        v_new = v[:, None] + v[None, :]
        k = torch.searchsorted(grid.classEdges, v_new.flatten()) - 1
        k = torch.clamp(k, 0, nclasses - 1).reshape(nclasses, nclasses)

        return k.cpu().numpy()

    def _precompute_geometry(self, grid):
        """Pre-compute geometric properties."""
        v = grid.meanParticuleVolume.cpu().numpy()

        self.radii = (3 * v / (4 * np.pi * RHO_SOOT)) ** (1 / 3)
        self.surfaces = 4 * np.pi * self.radii ** 2

        kn = lambda_g / self.radii
        self.cc = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))

    def coagulation_dt_limit(self, grid, CFL=0.2):
        """
        Estimate time step limit for coagulation stability.

        Returns:
            dt_limit: Scalar tensor
        """
        ns = grid.cells.ns.cpu().numpy()
        T = grid.cells.T.cpu().numpy()
        mu_gas = np.full(ns.shape[0], mu)

        dt_min = 1e10

        for cell_idx in range(ns.shape[0]):
            K = _compute_brownian_kernel_matrix(T[cell_idx], mu_gas[cell_idx])
            loss_rate = K @ ns[cell_idx]
            tau = np.where(loss_rate > 1e-30, 1.0 / loss_rate, 1e10)
            dt_min = min(dt_min, np.min(tau))

        return torch.tensor(CFL * dt_min, device=grid.device)

    def computeIntegratedNs(self, grid, dt):
        """
        Main integration routine - parallel CPU execution with scipy.
        Nucleation is now integrated in the BDF system.

        Args:
            grid: Grid object with cells.ns (ncells, nclasses)
            dt: Time step

        Returns:
            ns_new: Updated particle number densities (ncells, nclasses)
        """
        # Convert to numpy
        ns0_all = grid.cells.ns.cpu().numpy()
        T_all = grid.cells.T.cpu().numpy()
        rho_all = grid.cells.rho.cpu().numpy()
        Ys_all = grid.cells.Ys.cpu().numpy()
        mu_gas_all = np.full(ns0_all.shape[0], mu)

        # Prepare tasks for parallel processing
        tasks = [
            (i, ns0_all[i], T_all[i], rho_all[i], Ys_all[i], mu_gas_all[i],
             dt, self.indices, self.v_nuc)
            for i in range(self.ncells)
        ]

        # Parallel integration
        results = self.pool.map(
            _integrate_cell_soot,
            tasks,
            chunksize=max(1, self.ncells // (4 * self.nproc))
        )

        # Reconstruct torch tensor
        device = grid.cells.ns.device
        dtype = grid.cells.ns.dtype

        ns_final = torch.empty_like(grid.cells.ns)
        for cell_idx, ns in results:
            ns_final[cell_idx] = torch.from_numpy(ns)

        return ns_final.to(device=device, dtype=dtype)