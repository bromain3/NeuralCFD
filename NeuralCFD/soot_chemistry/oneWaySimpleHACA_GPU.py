"""
oneWayHACA 100% GPU-optimized implementation with RK45 integration.
All operations vectorized for PyTorch GPU execution.

Features:
- Fully batched nucleation, surface growth, oxidation, and coagulation
- GPU-accelerated Brownian coagulation kernel computation
- RK45 integration for stiff population balance equations
- No CPU/numpy conversions in integration loop
- Conservative coagulation with vectorized mapping

Author: Optimized version for GPU execution
"""

import torch
from NeuralCFD.numerical_integrators.RK45 import RK45Integrator
from NeuralCFD.soot_chemistry.DummySoot import DummySoot


# Physical constants
R = 8.314  # J/(mol·K)
kB = 1.380649e-23  # J/K
RHO_SOOT = 1800.0  # kg/m³
lambda_g = 68e-9  # Mean free path (m)
mu = 1.8e-5  # Dynamic viscosity (Pa·s)


class oneWaySimpleHACA_GPU(DummySoot):
    """
    GPU-optimized one-way HACA soot model.
    All computations performed on GPU with batched operations.
    """

    def __init__(self, grid, thermo, C2H2name='C2H2', O2name='O2', OHname='OH', Oname='O',
                 A_HACA=1.0e5, n_HACA=0.0, Ea_HACA=1.5e5, alpha=1.0):
        """
        Initialize GPU-optimized HACA model.

        Args:
            grid: Grid object containing particle size distribution
            thermo: Thermodynamic properties object
            integrator_class: RK45Integrator or similar class
            C2H2name, O2name, OHname: Species names
            A_nuc, EA_nuc: Nucleation Arrhenius parameters
            C2H2order: Reaction order for C2H2
            k_OH, k_O2: Oxidation rate constants
            A_HACA, n_HACA, Ea_HACA: HACA growth parameters
            alpha: HACA efficiency factor
            rtol, atol: Integration tolerances
            verbose: Print diagnostic info
        """
        self.device = grid.device
        self.dtype = grid.meanParticuleVolume.dtype

        # Species indices
        self.C2H2idx = thermo.thermoProp.species.index(C2H2name)
        self.OHidx = thermo.thermoProp.species.index(OHname)
        self.O2idx = thermo.thermoProp.species.index(O2name)
        self.Oidx = thermo.thermoProp.species.index(Oname)

        # Kinetic constants (convert to tensors)
        self.A_HACA = torch.tensor(A_HACA, device=self.device, dtype=self.dtype)
        self.n_HACA = n_HACA
        self.Ea_HACA = torch.tensor(Ea_HACA, device=self.device, dtype=self.dtype)
        self.alpha = alpha

        # Constants as tensors
        self.R = torch.tensor(R, device=self.device, dtype=self.dtype)
        self.kB = torch.tensor(kB, device=self.device, dtype=self.dtype)
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=self.device, dtype=self.dtype)
        self.lambda_g = torch.tensor(lambda_g, device=self.device, dtype=self.dtype)
        self.mu = torch.tensor(mu, device=self.device, dtype=self.dtype)

        # Coagulation map
        self.coag_map = self._build_conservative_coag_map(grid)

        # Pre-compute geometry on GPU
        self._precompute_geometry(grid)

        # Store grid info
        self.nclasses = grid.nclasses
        self.sectionSize = grid.sectionSize


    def _build_conservative_coag_map(self, grid):
        """Vectorized coagulation map construction on GPU."""
        nclasses = grid.nclasses
        v = grid.meanParticuleVolume  # (nclasses,) on GPU

        # Broadcasting: v_new[i,j] = v[i] + v[j]
        v_new = v[:, None] + v[None, :]  # (nclasses, nclasses)

        # Vectorized search
        k = torch.searchsorted(grid.classEdges, v_new.flatten()) - 1
        k = torch.clamp(k, 0, nclasses - 1).reshape(nclasses, nclasses)

        return k

    def _precompute_geometry(self, grid):
        """Pre-compute geometric properties on GPU."""
        v = grid.meanParticuleVolume  # (nclasses,) on GPU

        # Radii and surfaces
        self.radii = (3 * v / (4 * torch.pi * self.RHO_SOOT)) ** (1 / 3)  # (nclasses,)
        self.surfaces = 4 * torch.pi * self.radii ** 2  # (nclasses,)

        # Cunningham correction (independent of T at first order)
        kn = self.lambda_g / self.radii
        self.cc = 1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn))  # (nclasses,)

    def nucleation_Leung(self, grid): # Leung model. The incipient particles diameter is approximately 1.24nm
        """
        Nucleation source term (batched).

        Returns:
            dndt: (ncells, nclasses) tensor
        """

        # C2H2 concentration
        C_C2H2 = grid.cells.Ys[:, self.C2H2idx] * grid.cells.rho  # (ncells,) #TODO: directly extract Q where concentration is stored
        # Nucleation rate
        k1 = 0.1e5 * torch.exp(-21_000 / grid.cells.T)  # 1/s

        R_nuc = k1 * C_C2H2
        C2H2_loss = -R_nuc
        H2_gain = R_nuc
        # Mass of nucleated particle
        v_nuc = grid.meanParticuleVolume[0]
        m_p = self.RHO_SOOT * v_nuc
        # Number density rate (only first section)
        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, 0] = R_nuc / m_p
        return dndt

    def _compute_R_HACA(self, grid):
        """
        HACA growth rate (batched).

        Returns:
            R_HACA: (ncells,) tensor
        """
        T = grid.cells.T  # (ncells,)
        k_HACA = self.A_HACA * T ** self.n_HACA * torch.exp(-self.Ea_HACA / (self.R * T))
        return k_HACA * grid.cells.Ys[:, self.C2H2idx] * self.alpha

    def _compute_brownian_kernel_matrix(self, grid):
        """
        GPU-accelerated Brownian coagulation kernel computation.

        Returns:
            K: (ncells, nclasses, nclasses) tensor on GPU
        """
        ncells = grid.cells.ns.shape[0]
        T = grid.cells.T  # (ncells,)
        # Use gas viscosity if available, else constant
        if hasattr(grid.cells, 'mu'):
            mu_gas = grid.cells.mu  # (ncells,)
        else:
            mu_gas = self.mu.expand(ncells)
        # Broadcasting: (ncells, 1, 1) × (1, nclasses, nclasses)
        ri = self.radii[None, :, None]  # (1, nclasses, 1)
        rj = self.radii[None, None, :]  # (1, 1, nclasses)
        cc_i = self.cc[None, :, None]
        cc_j = self.cc[None, None, :]
        # Prefactor: 2*kB*T / (3*mu)
        prefactor = (2 * self.kB * T / (3 * mu_gas))[:, None, None]  # (ncells, 1, 1)
        # Kernel (fully vectorized)
        K = prefactor * (ri + rj) * (cc_i / ri + cc_j / rj)  # (ncells, nclasses, nclasses)
        return K

    def _surface_growth_batched(self, ns, G):
        """
        Vectorized surface growth for all cells.

        Args:
            ns: (ncells, nclasses) particle number densities
            G: (ncells, nclasses) growth rates

        Returns:
            dndt: (ncells, nclasses) rate of change
        """
        flux = G * ns  # (ncells, nclasses)
        dndt = torch.zeros_like(ns)
        # Incoming flux to section i from i-1
        dndt[:, 1:] += flux[:, :-1] / self.sectionSize[1:]
        # Outgoing flux from section i to i+1
        dndt[:, :-1] -= flux[:, :-1] / self.sectionSize[:-1]
        return dndt



    def _oxidation_batched(self, ns, L):
        """
        Vectorized oxidation for all cells.

        Args:
            ns: (ncells, nclasses) particle number densities
            L: (ncells, nclasses) oxidation rates

        Returns:
            dndt: (ncells, nclasses) rate of change
        """
        flux = L * ns  # (ncells, nclasses)
        dndt = torch.zeros_like(ns)
        # Incoming flux to section i from i+1 (reverse direction)
        dndt[:, :-1] += flux[:, 1:] / self.sectionSize[:-1]
        # Outgoing flux from section i to i-1
        dndt[:, 1:] -= flux[:, 1:] / self.sectionSize[1:]
        # DESTRUCTION: Class 0 particles shrink below minimum size and are destroyed
        # They leave class 0 but don't go anywhere → net destruction
        dndt[:, 0] -= flux[:, 0] / self.sectionSize[0]
        return dndt

    def _coagulation_batched(self, ns, K, coag_map):
        """
        Version 3D 100% vectorisée (sans boucle for).
        """
        ncells, nclasses = ns.shape
        # 1. Calcul des produits extérieurs (ni * nj)
        ns_i = ns.unsqueeze(2)
        ns_j = ns.unsqueeze(1)
        product = ns_i * ns_j  # (ncells, nclasses, nclasses)
        # 2. Application du noyau et du facteur de symétrie
        mask_diag = torch.eye(nclasses, device=self.device, dtype=self.dtype)
        symmetry_factor = 0.5 * mask_diag + (1.0 - mask_diag)
        rate = K * product * symmetry_factor  # (ncells, nclasses, nclasses)
        # 3. Termes de perte (Loss)
        # On somme sur j pour avoir la perte de i, et on ajoute la perte de j
        loss_i = rate.sum(dim=2)
        loss_j = rate.sum(dim=1)
        # On retire la diagonale une fois car elle est comptée deux fois dans i+j
        diag_rates = torch.diagonal(rate, dim1=1, dim2=2)
        total_loss = loss_i + loss_j - diag_rates
        # 4. Termes de gain (Gain) - Transformation 3D ici
        dndt = -total_loss
        # Aplatir les taux : (ncells, nclasses * nclasses)
        rate_flat = rate.view(ncells, -1)
        # Préparer la map en 3D : (ncells, nclasses * nclasses)
        # expand() crée une vue virtuelle sans copier les données
        coag_map_3d = coag_map.flatten().expand(ncells, -1)
        # Un seul scatter_add pour toutes les cellules d'un coup
        dndt.scatter_add_(1, coag_map_3d, rate_flat)
        return dndt

    def coagulation_dt_limit(self, grid, CFL=0.2):
        """
        Estimate time step limit for coagulation stability.

        Returns:
            dt_limit: Scalar tensor
        """
        ns = grid.cells.ns
        K = self._compute_brownian_kernel_matrix(grid)
        # Loss rate: sum_j K_ij * n_j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)
        # Characteristic time (avoid division by zero)
        tau = torch.where(loss_rate > 1e-30, 1.0 / loss_rate, 1e10)
        dt_limit = CFL * torch.min(tau)
        return dt_limit

    def computeIntegratedNs(self, grid, dt):
        """
        Main integration routine - 100% GPU execution with RK45.

        Args:
            grid: Grid object with cells.ns (ncells, nclasses)
            dt: Time step

        Returns:
            ns_new: Updated particle number densities (ncells, nclasses)
        """
        # Pre-compute all rate parameters on GPU
        R_HACA = self._compute_R_HACA(grid)  # (ncells,)
        K = self._compute_brownian_kernel_matrix(grid)  # (ncells, nclasses, nclasses)
        # Surface growth rates G[cell, class]
        G = self.surfaces[None, :] * R_HACA[:, None]  # (ncells, nclasses)
        # Oxidation rates L[cell, class]
        MW_OH = 17e-3  # kg/mol
        MW_O = 16e-3
        C_OH = grid.cells.rho * grid.cells.Ys[:, self.OHidx] / MW_OH
        C_O = grid.cells.rho * grid.cells.Ys[:, self.Oidx] / MW_O
        k = 2.0e7 * torch.exp(-87864 / (8.314 * grid.cells.T))
        oxyd_rate = k * (self.alpha * C_OH + C_O)
        L = self.surfaces[None, :] * oxyd_rate[:, None]  # (ncells, nclasses)
        # Store for RHS function
        self._G = G
        self._L = L
        self._K = K
        # Define RHS function for integrator
        def rhs_soot(ns, t):
            """
            Right-hand side for population balance equations.

            Args:
                ns: (ncells, nclasses) current state
                t: time (unused, autonomous system)

            Returns:
                dndt: (ncells, nclasses) rate of change
            """
            # Ensure non-negative
            ns = torch.clamp(ns, min=0.0)
            # Compute all terms
            dn_growth = self._surface_growth_batched(ns, self._G)
            dn_oxyd = self._oxidation_batched(ns, self._L)
            dn_coag = self._coagulation_batched(ns, self._K, self.coag_map)
            dn_nucleation = self.nucleation_Leung(grid)
            #print(dn_growth + dn_oxyd + dn_coag + dn_nucleation)
            return dn_growth + dn_oxyd + dn_coag + dn_nucleation

        # Create integrator instance
        integrator = RK45Integrator(  #TODO: optimize by not creating an instance at each time but just updating the rhs
            rhs_soot,
            device=self.device,
            dtype=self.dtype,
            #rtol=self.rtol,
            #atol=self.atol,
            #verbose=True
        )

        # Initial condition
        ns0 = grid.cells.ns.clone()

        # Integrate over [0, dt]
        t_vals, ns_vals = integrator.integrate(ns0, t_span=(0.0, dt))
        ns_final = ns_vals[-1]  # Last time point

        # Add nucleation (can be integrated separately or as source term)

        # Ensure non-negative
        ns_total = torch.clamp(ns_final, min=0.0)


        return ns_total
