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


"""
Detailed Chemical Mechanism for Soot Oxidation
================================================

Multi-site surface chemistry model based on:
- Frenklach et al. (2018) - Multi-site kinetics
- Edwards & Frenklach (2014) - OH elementary pathways  
- Singh et al. (2015) - O2 oxidation of graphene edges
- Galimova et al. (2018) - Five-member ring oxidation by O

Elementary surface reactions:
1. H-abstraction creating radical sites
2. O2 attack on radical sites forming oxyradicals
3. OH attack on radical sites forming oxyradicals
4. Oxyradical decomposition releasing CO
5. Five-member ring formation and oxidation
6. O atom oxidation of embedded rings

Surface species tracked:
- C-H (zigzag and armchair edges)
- C• (radical sites)
- C-O• (oxyradicals)
- C5 (embedded five-member rings)
"""


class twoWaySimpleHACA_GPU(DummySoot):
    """
    GPU-optimized one-way HACA soot model.
    All computations performed on GPU with batched operations.
    """

    def __init__(self, grid, thermo, C2H2name='C2H2', O2name='O2', OHname='OH', Oname='O', Hname='H', H2name='H2', COname='CO',
                 k_OH=1.0e4, k_O2=2.0e2,
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
        self.Hidx = thermo.thermoProp.species.index(Hname)
        self.H2idx = thermo.thermoProp.species.index(H2name)
        self.COidx = thermo.thermoProp.species.index(COname)

        # Kinetic constants (convert to tensors)
        self.k_OH = torch.tensor(k_OH, device=self.device, dtype=self.dtype)
        self.k_O2 = torch.tensor(k_O2, device=self.device, dtype=self.dtype)
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

    def init_surface_sites(self, grid):
        """
        Initialize surface site densities.

        Site types:
        - CH_zigzag: C-H on zigzag edges
        - CH_armchair: C-H on armchair edges
        - Crad: C• radical sites
        - COrad: C-O• oxyradical sites
        - C5: Embedded five-member rings
        """
        ncells, nclasses = grid.cells.ns.shape

        # Site density [sites/m²]
        # Typical: 1-3 × 10^19 sites/m² (0.3-0.5 nm² per site)
        site_density_total = 2.0e19  # sites/m²

        # Initial distribution (nascent soot is mostly C-H)
        self.sites = {
            'CH_zigzag': torch.ones((ncells, nclasses), device=self.device,
                                    dtype=self.dtype) * 0.6 * site_density_total,
            'CH_armchair': torch.ones((ncells, nclasses), device=self.device,
                                      dtype=self.dtype) * 0.3 * site_density_total,
            'Crad': torch.ones((ncells, nclasses), device=self.device,
                               dtype=self.dtype) * 0.09 * site_density_total,
            'COrad': torch.zeros((ncells, nclasses), device=self.device, dtype=self.dtype),
            'C5': torch.ones((ncells, nclasses), device=self.device, dtype=self.dtype) * 0.01 * site_density_total,
        }

    def compute_elementary_rates(self, grid):
        """
        Compute all elementary reaction rates.

        Returns:
            rates: dict of reaction rates
            gas_sources: gas phase source terms
        """
        T = grid.cells.T  # (ncells,)
        rho = grid.cells.rho

        # Gas concentrations [molecules/m³]
        N_H = grid.cells.Ys[:, self.Hidx] * rho / 1.008e-3 * self.N_A
        N_OH = grid.cells.Ys[:, self.OHidx] * rho / 17.008e-3 * self.N_A
        N_O2 = grid.cells.Ys[:, self.O2idx] * rho / 32.0e-3 * self.N_A
        N_O = grid.cells.Ys[:, self.Oidx] * rho / 16.0e-3 * self.N_A
        N_H2 = grid.cells.Ys[:, self.H2idx] * rho / 2.016e-3 * self.N_A
        N_C2H2 = grid.cells.Ys[:, self.C2H2idx] * rho / 26.04e-3 * self.N_A

        # Surface areas [m²]
        N_particles = grid.cells.ns * rho  # [particles/m³]
        S = self.surfaces[None, :] * N_particles  # [m²/m³]

        rates = {}

        # ================================================================
        # REACTION 1: H-ABSTRACTION (Creating radical sites)
        # ================================================================
        # C-H + H → C• + H2
        # Rate from Frenklach: k = A * T^n * exp(-Ea/RT)

        # Zigzag edges (more reactive)
        k1_zigzag = 4.2e7 * T[:, None] ** 1.5 * torch.exp(-8000.0 / (self.R * T[:, None]))  # [m³/(molecule·s)]
        rates['H_abstraction_zigzag'] = k1_zigzag * self.sites['CH_zigzag'] * S * N_H[:, None]  # [sites/m³/s]

        # Armchair edges (less reactive)
        k1_armchair = 2.0e7 * T[:, None] ** 1.5 * torch.exp(-10000.0 / (self.R * T[:, None]))
        rates['H_abstraction_armchair'] = k1_armchair * self.sites['CH_armchair'] * S * N_H[:, None]

        # ================================================================
        # REACTION 2: H-ABSTRACTION BY OH (faster)
        # ================================================================
        # C-H + OH → C• + H2O
        k2_zigzag = 1.0e8 * T[:, None] ** 1.0 * torch.exp(-3000.0 / (self.R * T[:, None]))
        rates['OH_abstraction_zigzag'] = k2_zigzag * self.sites['CH_zigzag'] * S * N_OH[:, None]

        k2_armchair = 5.0e7 * T[:, None] ** 1.0 * torch.exp(-4000.0 / (self.R * T[:, None]))
        rates['OH_abstraction_armchair'] = k2_armchair * self.sites['CH_armchair'] * S * N_OH[:, None]

        # ================================================================
        # REACTION 3: O2 ATTACK ON RADICALS (Forming oxyradicals)
        # ================================================================
        # C• + O2 → C-O-O• (peroxy intermediate, fast)
        # C-O-O• → C-O• + O (fast rearrangement)
        # Net: C• + O2 → C-O• + O

        # From Singh et al. (2015) - activation energy ~0 for radical + O2
        k3 = 3.0e8 * torch.exp(-2000.0 / (self.R * T[:, None]))  # [m³/(molecule·s)]
        rates['O2_oxidation'] = k3 * self.sites['Crad'] * S * N_O2[:, None]

        # ================================================================
        # REACTION 4: OH ATTACK ON RADICALS (Direct oxyradical formation)
        # ================================================================
        # C• + OH → C-O• + H
        # From Edwards & Frenklach (2014): k ~ 1e14 cm³/(mol·s) = 1.66e-10 m³/(molecule·s)
        # Nearly barrierless
        k4 = 1.66e-10  # Temperature-independent (very small Ea)
        rates['OH_oxidation'] = k4 * self.sites['Crad'] * S * N_OH[:, None]

        # ================================================================
        # REACTION 5: OXYRADICAL DECOMPOSITION (CO release)
        # ================================================================
        # C-O• → CO + C• (new radical site)
        # This is the rate-limiting step at low-moderate T
        # From Edwards thesis: multiple pathways, average Ea ~ 140-180 kJ/mol

        k5 = 1.0e13 * torch.exp(-150000.0 / (self.R * T[:, None]))  # [1/s]
        rates['oxyradical_decomp'] = k5 * self.sites['COrad'] * S

        # ================================================================
        # REACTION 6: FIVE-MEMBER RING FORMATION
        # ================================================================
        # C-O• → C5 (embedded ring) + CO
        # Competing pathway to decomposition
        # Leads to less reactive surface

        k6 = 5.0e12 * torch.exp(-160000.0 / (self.R * T[:, None]))  # [1/s]
        rates['C5_formation'] = k6 * self.sites['COrad'] * S

        # ================================================================
        # REACTION 7: FIVE-MEMBER RING OXIDATION BY O
        # ================================================================
        # C5 + O → CO + C• (reformed radical)
        # From Galimova et al. (2018): Only O atoms can break these rings
        # Activation energy ~120 kJ/mol

        k7 = 5.0e8 * torch.exp(-120000.0 / (self.R * T[:, None]))  # [m³/(molecule·s)]
        rates['C5_oxidation'] = k7 * self.sites['C5'] * S * N_O[:, None]

        # ================================================================
        # REACTION 8: RADICAL RECOMBINATION WITH H2
        # ================================================================
        # C• + H2 → C-H + H (reverse of abstraction)
        # Slower than abstraction due to H2 being less reactive

        k8 = 1.0e6 * T[:, None] ** 1.5 * torch.exp(-30000.0 / (self.R * T[:, None]))
        rates['H2_addition_zigzag'] = k8 * self.sites['Crad'] * S * N_H2[:, None] * 0.6  # 60% zigzag
        rates['H2_addition_armchair'] = k8 * self.sites['Crad'] * S * N_H2[:, None] * 0.4  # 40% armchair

        # ================================================================
        # REACTION 9: C2H2 ADDITION TO RADICALS (Growth, not oxidation)
        # ================================================================
        # C• + C2H2 → C-C2H2• (HACA growth)
        # Included for completeness as it competes with oxidation

        k9 = 8.0e7 * T[:, None] ** 0.5 * torch.exp(-10000.0 / (self.R * T[:, None]))
        rates['C2H2_addition'] = k9 * self.sites['Crad'] * S * N_C2H2[:, None]

        return rates

    def update_surface_sites(self, grid, rates, dt):
        """
        Update surface site densities based on reaction rates.

        Uses explicit Euler integration (can be improved with RK45).
        """
        # Site conservation equations:

        # d[CH_zigzag]/dt
        dCH_zig = (
                - rates['H_abstraction_zigzag']  # Loss: H abstraction
                - rates['OH_abstraction_zigzag']  # Loss: OH abstraction
                + rates['H2_addition_zigzag']  # Gain: H2 addition
        )

        # d[CH_armchair]/dt
        dCH_arm = (
                - rates['H_abstraction_armchair']
                - rates['OH_abstraction_armchair']
                + rates['H2_addition_armchair']
        )

        # d[Crad]/dt
        dCrad = (
                + rates['H_abstraction_zigzag']  # Gain: from H abstraction
                + rates['H_abstraction_armchair']
                + rates['OH_abstraction_zigzag']
                + rates['OH_abstraction_armchair']
                + rates['oxyradical_decomp']  # Gain: from CO release
                + rates['C5_oxidation']  # Gain: from C5 oxidation
                - rates['O2_oxidation']  # Loss: O2 attack
                - rates['OH_oxidation']  # Loss: OH attack
                - rates['H2_addition_zigzag']  # Loss: H2 addition
                - rates['H2_addition_armchair']
                - rates['C2H2_addition']  # Loss: C2H2 growth
        )

        # d[COrad]/dt
        dCOrad = (
                + rates['O2_oxidation']  # Gain: from O2
                + rates['OH_oxidation']  # Gain: from OH
                - rates['oxyradical_decomp']  # Loss: decomposition
                - rates['C5_formation']  # Loss: ring formation
        )

        # d[C5]/dt
        dC5 = (
                + rates['C5_formation']  # Gain: from oxyradical
                - rates['C5_oxidation']  # Loss: O oxidation
        )

        # Update with time step
        self.sites['CH_zigzag'] = torch.clamp(self.sites['CH_zigzag'] + dCH_zig * dt, min=0.0)
        self.sites['CH_armchair'] = torch.clamp(self.sites['CH_armchair'] + dCH_arm * dt, min=0.0)
        self.sites['Crad'] = torch.clamp(self.sites['Crad'] + dCrad * dt, min=0.0)
        self.sites['COrad'] = torch.clamp(self.sites['COrad'] + dCOrad * dt, min=0.0)
        self.sites['C5'] = torch.clamp(self.sites['C5'] + dC5 * dt, min=0.0)

    def compute_net_oxidation(self, grid, rates):
        """
        Compute net carbon removal rate (oxidation).

        Returns:
            R_oxid: (ncells, nclasses) oxidation rate [kmol C/m³/s]
            gas_sources: dict of gas species rates [kmol/m³/s]
        """
        # Net CO production = CO from oxyradical decomp + CO from C5 oxidation
        R_CO_sites = rates['oxyradical_decomp'] + rates['C5_formation'] + rates['C5_oxidation']  # [sites/m³/s]

        # Convert sites/s to kmol C/s
        # 1 site = 1 carbon atom
        R_CO = R_CO_sites / self.N_A * 1000  # [kmol/m³/s]

        # This is the net oxidation (carbon removal)
        R_oxid = R_CO

        # Gas phase sources
        rho = grid.cells.rho

        # Sum over all size classes for each cell
        R_CO_total = torch.sum(R_CO, dim=1)
        R_O2_total = torch.sum(rates['O2_oxidation'], dim=1) / self.N_A * 1000
        R_OH_total = torch.sum(
            rates['OH_oxidation'] + rates['OH_abstraction_zigzag'] + rates['OH_abstraction_armchair'],
            dim=1) / self.N_A * 1000
        R_O_total = torch.sum(rates['C5_oxidation'], dim=1) / self.N_A * 1000
        R_H_total = torch.sum(rates['H_abstraction_zigzag'] + rates['H_abstraction_armchair'],
                              dim=1) / self.N_A * 1000
        R_H2_total = torch.sum(rates['H2_addition_zigzag'] + rates['H2_addition_armchair'], dim=1) / self.N_A * 1000
        R_C2H2_total = torch.sum(rates['C2H2_addition'], dim=1) / self.N_A * 1000

        gas_sources = {
            'CO': R_CO_total / rho,  # Produced
            'O2': -R_O2_total / rho,  # Consumed
            'OH': -R_OH_total / rho,  # Consumed
            'O': -R_O_total / rho + R_O2_total / rho,  # Consumed by C5, produced by O2 reaction
            'H': -R_H_total / rho + R_OH_total / rho,  # Net depends on pathway
            'H2': -R_H2_total / rho + R_H_total / rho,  # Produced by abstraction, consumed by addition
            'H2O': R_OH_total / rho,  # Produced from OH abstraction (subset)
            'C2H2': -R_C2H2_total / rho,  # Consumed (growth, not oxidation)
        }

        return R_oxid, gas_sources

    def integrate_oxidation(self, grid, dt):
        """
        Main integration routine for detailed chemistry.

        Args:
            grid: Grid object
            dt: time step

        Returns:
            L: (ncells, nclasses) volumetric loss rate [m³/s per particle]
            diagnostics: dict with detailed info
            gas_sources: gas phase feedback
        """
        # Compute reaction rates
        rates = self.compute_elementary_rates(grid)

        # Update surface sites
        self.update_surface_sites(grid, rates, dt)

        # Compute net oxidation
        R_oxid, gas_sources = self.compute_net_oxidation(grid, rates)

        # Convert to volumetric loss rate
        rho = grid.cells.rho
        N_particles = grid.cells.ns * rho
        L = -R_oxid * self.M_C / self.RHO_SOOT / (N_particles + 1e-30)  # [m³/s per particle]

        # Diagnostics
        diagnostics = {
            'R_oxid': R_oxid,
            'rates': rates,
            'sites': self.sites.copy(),
            'radical_fraction': self.sites['Crad'] / (
                        self.sites['CH_zigzag'] + self.sites['CH_armchair'] + self.sites['Crad'] + 1e-30),
            'C5_fraction': self.sites['C5'] / (
                        self.sites['CH_zigzag'] + self.sites['CH_armchair'] + self.sites['Crad'] + self.sites[
                    'C5'] + 1e-30),
        }

        return L, diagnostics, gas_sources

# ================================================================
# SIMPLIFIED INTERFACE FOR SECTIONAL MODEL
# ================================================================

def detailed_oxidation_batched(oxidation_model, grid, ns, dt):
    """
    Wrapper for sectional integration with detailed chemistry.

    Args:
        oxidation_model: DetailedSootOxidationChemistry instance
        grid: Grid object
        ns: (ncells, nclasses) particle number densities
        dt: time step

    Returns:
        dndt: (ncells, nclasses) rate of change
        diagnostics: detailed chemistry diagnostics
        gas_sources: gas phase feedback
    """
    # Compute oxidation with detailed chemistry
    L, diagnostics, gas_sources = oxidation_model.integrate_oxidation(grid, dt)

    # Apply sectional method
    flux = L * ns  # [m³/kg/s]
    dndt = torch.zeros_like(ns)

    # Sectional transport
    sectionSize = grid.sectionSize if hasattr(grid, 'sectionSize') else torch.diff(grid.classEdges)
    dndt[:, 1:] -= flux[:, 1:] / sectionSize[1:]
    dndt[:, :-1] += flux[:, 1:] / sectionSize[:-1]

    # Particle destruction at minimum size
    dndt[:, 0] -= flux[:, 0] / sectionSize[0]

    return dndt, diagnostics, gas_sources


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
        oxyd_rate = (self.k_OH * self.alpha * grid.cells.Ys[:, self.OHidx] +
                     self.k_O2 * grid.cells.Ys[:, self.O2idx]) * grid.cells.rho
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
