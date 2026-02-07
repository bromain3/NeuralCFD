"""
oneWayLL - Modèle de suie Leung & Lindstedt (1991) 100% GPU-optimized
Implémentation exacte des équations de SootLib avec coagulation

ÉQUATIONS IMPLÉMENTÉES:
- Nucléation: Leung & Lindstedt (1991)
- Croissance: Leung & Lindstedt (1991)
- Oxydation: Leung & Lindstedt (1991)
- Coagulation: Brownian (comme oneWayHACA)

UNITÉS:
- Entrée: Y (fraction massique), T (K), P (Pa), ρ (kg/m³)
- Interne: Concentrations en mol/m³ pour cohérence avec Arrhenius
- Sortie: ns (#/m³)

Author: Implémentation rigoureuse SootLib
"""

import torch
from NeuralCFD.numerical_integrators.RK45 import RK45Integrator
from NeuralCFD.soot_chemistry.DummySoot import DummySoot


# ============== CONSTANTES PHYSIQUES ==============
R_SI = 8.314          # J/(mol·K)
R_CAL = 1.987         # cal/(mol·K)
CAL_TO_J = 4.184      # J/cal
kB = 1.380649e-23     # J/K
Na = 6.022e23         # #/mol
RHO_SOOT = 1800.0     # kg/m³ (densité suie)
lambda_g = 68e-9      # m (libre parcours moyen gaz)
mu = 1.8e-5           # Pa·s (viscosité dynamique)

# ============== MASSES MOLAIRES (kg/mol) ==============
MW_C = 12e-3
MW_C2H2 = 26e-3
MW_O2 = 32e-3
MW_H2 = 2e-3
MW_CO = 28e-3


class oneWay_Leung_Lindset(DummySoot):
    """
    Modèle Leung & Lindstedt (1991) - One-way coupling

    Nucléation:  C2H2 --> 2C(s) + H2
    Croissance:  C2H2 + nC(s) --> (n+2)C(s) + H2
    Oxydation:   C(s) + 0.5O2 --> CO
    """

    def __init__(self, grid, thermo, C2H2name='C2H2', O2name='O2'):
        """
        Initialisation du modèle LL

        Args:
            grid: grille de calcul
            thermo: propriétés thermodynamiques
            C2H2name: nom de l'espèce C2H2
            O2name: nom de l'espèce O2
            C_min: nombre d'atomes de carbone dans particule nucléée
        """
        self.device = grid.device
        self.dtype = grid.meanParticuleVolume.dtype

        # ============== INDICES DES ESPÈCES ==============
        self.C2H2idx = thermo.thermoProp.species.index(C2H2name)
        self.O2idx = thermo.thermoProp.species.index(O2name)

        # ============== CONSTANTES PHYSIQUES (tenseurs GPU) ==============
        self.R_SI = torch.tensor(R_SI, device=self.device, dtype=self.dtype)
        self.R_CAL = torch.tensor(R_CAL, device=self.device, dtype=self.dtype)
        self.kB = torch.tensor(kB, device=self.device, dtype=self.dtype)
        self.Na = torch.tensor(Na, device=self.device, dtype=self.dtype)
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=self.device, dtype=self.dtype)
        self.lambda_g = torch.tensor(lambda_g, device=self.device, dtype=self.dtype)
        self.mu = torch.tensor(mu, device=self.device, dtype=self.dtype)

        # ============== MASSES MOLAIRES (tenseurs GPU) ==============
        self.MW_C = torch.tensor(MW_C, device=self.device, dtype=self.dtype)
        self.MW_C2H2 = torch.tensor(MW_C2H2, device=self.device, dtype=self.dtype)
        self.MW_O2 = torch.tensor(MW_O2, device=self.device, dtype=self.dtype)
        self.MW_H2 = torch.tensor(MW_H2, device=self.device, dtype=self.dtype)
        self.MW_CO = torch.tensor(MW_CO, device=self.device, dtype=self.dtype)

        # ============== PARAMÈTRES DE NUCLÉATION ==============

        # Volume de la particule nucléée (m³)
        # self.v_nuc = (π/6) × d³
        self.v_nuc = grid.meanParticuleVolume[0].detach().clone()

        # Masse de la particule nucléée (kg)
        self.m_nuc = self.v_nuc * self.RHO_SOOT

        # Masse d'une particule dans une section donnée (kg)
        self.m_class = grid.meanParticuleVolume * self.RHO_SOOT

        # Nb de carbonnes de la particule nucléée
        # m_nuc = C_min × MW_C / Na
        self.C_min = self.m_nuc * self.Na / self.MW_C

        # ============== COAGULATION ==============
        self.coag_map = self._build_conservative_coag_map(grid)
        self._precompute_geometry(grid)

        # ============== GRID INFO ==============
        self.nclasses = grid.nclasses

    def _build_conservative_coag_map(self, grid):
        """Construction vectorisée de la carte de coagulation sur GPU."""
        nclasses = grid.nclasses
        v = grid.meanParticuleVolume  # (nclasses,) volumes en m³

        # v_new[i,j] = v[i] + v[j]
        v_new = v[:, None] + v[None, :]  # (nclasses, nclasses)

        # Trouver la classe de destination
        k = torch.searchsorted(grid.classEdges, v_new.flatten()) - 1
        k = torch.clamp(k, 0, nclasses - 1).reshape(nclasses, nclasses)

        return k


    def _precompute_geometry(self, grid):
        """Précalcul des propriétés géométriques sur GPU."""
        v = grid.meanParticuleVolume  # (nclasses,) volumes en m³

        # Rayons des particules: r = (3V/4π)^(1/3)
        self.radii = (3 * v / (4 * torch.pi)) ** (1 / 3)  # (nclasses,)

        # Surface des particules: S = 4πr²
        self.surfaces = 4 * torch.pi * self.radii ** 2  # (nclasses,)

        # Correction de Cunningham
        kn = self.lambda_g / self.radii
        self.cc = 1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn))  # (nclasses,)


    # ==================== NUCLÉATION ====================
    def nucleation_LL(self, grid):
        """
        Nucléation de Leung & Lindstedt (1991)

        Équation SootLib:
        Rnuc = 0.1×10^5 × exp(-21100/T) × [C2H2] × 2Na/Cmin

        Unités:
        - [C2H2] en mol/m³ (concentration molaire)
        - Rnuc en #/(m³·s)

        Retourne:
            dndt: (ncells, nclasses) - source de nucléation en #/(m³·s)
        """
        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_C2H2 = grid.cells.rho * grid.cells.Ys[:, self.C2H2idx] / self.MW_C2H2 / 1000  # (ncells,)

        # Constante de vitesse (d'après code SootLib)
        k_nuc = 0.1e5 * torch.exp(-21_100.0 / grid.cells.T)  # m³/(kmol·s)  # constante originelle de Leunget Lindset
        #k_nuc = 0.1e5 * torch.exp(-41_100.0 / grid.cells.T)  # m³/(kmol·s)  # constante plus récente

        # Taux de nucléation (kmol/m³/s)
        R_nuc_molar = k_nuc * C_C2H2  # kmol/(m³·s)

        # Convertir en #/m³/s avec facteur 2Na/Cmin
        # D'après SootLib: return Rnuc * 2 * Na / state.cMin car Na est en #/kmol
        R_nuc = R_nuc_molar * 2.0 * (self.Na * 1000) / self.C_min  # #/(m³·s)

        # Créer les particules dans la classe correspondant à v_nuc
        dndt = torch.zeros_like(grid.cells.ns)

        # Ajouter les particules nucléées à cette classe
        dndt[:, 0] = R_nuc

        return dndt


    # ==================== CROISSANCE SURFACIQUE ====================
    def growth_LL(self, ns, grid):
        """
        Croissance de Leung & Lindstedt (1991) - VERSION MOMENTS

        Équation SootLib (pour modèle à moments):
        rSoot = 0.6×10^4 × exp(-12100/T) × [C2H2] / √(A_m2m3) × 2 × MW_C

        Unités:
        - [C2H2] en mol/m³
        - rSoot en kg/(m²·s)
        - G = rSoot × S / ρ_s en m³/s

        Retourne:
            G: (ncells, nclasses) - taux de croissance volumique en m³/s
        """
        #TODO: adapter pour un modèle sectionnel en faisant un calcul par classe pour être plus précis

        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_C2H2 = grid.cells.rho * grid.cells.Ys[:, self.C2H2idx] / self.MW_C2H2 / 1000 # (ncells,)

        # Total number of particles per unit volume #/m3
        M0 = torch.sum(ns, dim=1)
        M1 = torch.sum(ns * self.m_class, dim=1)

        # soot surface available for reaction (m2_soot/m3_total)
        Am2m3 = torch.zeros_like(grid.cells.T)
        mask = M0 > 0.0
        Am2m3[mask] = torch.pi * torch.pow(6.0 / (torch.pi * self.RHO_SOOT) * M1[mask] / M0[mask], 2.0 / 3.0) * M0[mask]

        # soot growth rate (kg/m2/s)
        rSoot = torch.zeros_like(grid.cells.T)
        mask = Am2m3 > 0.0
        rSoot[mask] = 0.6e4 * torch.exp(-12_100.0 / grid.cells.T[mask]) * C_C2H2[mask] / torch.sqrt(Am2m3[mask]) * 2 * (MW_C * 1000)

        # soot volume growth rate per class (m3/s)
        growth_rate = rSoot[:, None] * grid.meanParticuleSurface[None, :] / self.RHO_SOOT


        ################### schéma numérique ###################
        # flux sortant de chaque classe
        flux_out = (growth_rate / grid.sectionSize[None, :]) * ns  # (#/m3/s)

        dndt = torch.zeros_like(ns)

        # entrée depuis la classe inférieure
        dndt[:, 1:] += flux_out[:, :-1]

        # sortie vers la classe supérieure
        dndt[:, :-1] -= flux_out[:, :-1]

        # no growth beyond last class so accumulation in the last class
        # flux_out[:, -1] = 0.0

        return dndt



    # ==================== OXYDATION ====================
    def oxidation_LL(self, ns, grid):
        """
        Oxydation de Leung & Lindstedt (1991)

        Équation SootLib:
        rOxi = 0.1×10^5 × √T × exp(-19680/T) × [O2] × MW_C

        Unités:
        - [O2] en mol/m³
        - rOxi en kg/(m²·s)
        - L = rOxi × S / ρ_s en m³/s

        Retourne:
            L: (ncells, nclasses) - taux d'oxydation volumique en m³/s
        """
        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_O2 = grid.cells.rho * grid.cells.Ys[:, self.O2idx] / self.MW_O2 / 1000 # (ncells,)

        # oxydation rate (kg/m2/s)
        rSoot = 0.1e5 * torch.sqrt(grid.cells.T) * torch.exp(-19_680.0 / grid.cells.T) * C_O2 * (MW_C * 1000)

        # soot volume growth rate per class (m3/s)
        ox_rate = rSoot[:, None] * grid.meanParticuleSurface[None, :] / self.RHO_SOOT

        ################### schéma numérique ###################
        # flux sortant de chaque classe
        flux_out = (ox_rate / grid.sectionSize[None, :]) * ns  # (#/m3/s)

        dndt = torch.zeros_like(ns)

        # Flux entrant dans section i depuis i+1 (sens inverse)
        dndt[:, :-1] += flux_out[:, 1:]

        # Flux sortant de section i vers i-1
        dndt[:, 1:] -= flux_out[:, 1:]

        # DESTRUCTION: Classe 0 disparaît sans aller nulle part
        dndt[:, 0] -= flux_out[:, 0]

        return dndt



    # ==================== COAGULATION ====================
    def _compute_brownian_kernel_matrix(self, grid):
        """
        Calcul du noyau de coagulation brownien (GPU)

        Formule de Fuchs (transition free molecular / continuum)
        """
        ncells = grid.ncells
        T = grid.cells.T  # (ncells,)

        # Viscosité
        if hasattr(grid.cells, 'mu'):
            mu_gas = grid.cells.mu  # (ncells,)
        else:
            mu_gas = self.mu.expand(ncells)

        # Broadcasting
        ri = self.radii[None, :, None]  # (1, nclasses, 1)
        rj = self.radii[None, None, :]  # (1, 1, nclasses)
        cc_i = self.cc[None, :, None]
        cc_j = self.cc[None, None, :]

        # Noyau de coagulation (approximation continuum simplifiée)
        # K = (2kBT/3μ) × (ri + rj) × (Cc_i/ri + Cc_j/rj)
        prefactor = (2 * self.kB * T / (3 * mu_gas))[:, None, None]  # (ncells, 1, 1)

        K = prefactor * (ri + rj) * (cc_i / ri + cc_j / rj)  # (ncells, nclasses, nclasses)

        return K

    def _coagulation_batched(self, ns, K, coag_map):
        """
        Coagulation 100% vectorisée

        Équation:
        dndt[i] = -Σ_j K[i,j] × ns[i] × ns[j] + Σ_(k,l→i) K[k,l] × ns[k] × ns[l]
        """
        ncells, nclasses = ns.shape

        # Produits ni × nj
        ns_i = ns.unsqueeze(2)  # (ncells, nclasses, 1)
        ns_j = ns.unsqueeze(1)  # (ncells, 1, nclasses)
        product = ns_i * ns_j   # (ncells, nclasses, nclasses)

        # Facteur de symétrie (1/2 pour i=j, 1 sinon)
        mask_diag = torch.eye(nclasses, device=self.device, dtype=self.dtype)
        symmetry_factor = 0.5 * mask_diag + (1.0 - mask_diag)

        # Taux de collision
        rate = K * product * symmetry_factor  # (ncells, nclasses, nclasses)

        # Termes de perte
        loss_i = rate.sum(dim=2)  # Σ_j K[i,j] × ns[i] × ns[j]
        loss_j = rate.sum(dim=1)  # Σ_i K[i,j] × ns[i] × ns[j]
        diag_rates = torch.diagonal(rate, dim1=1, dim2=2)

        total_loss = loss_i + loss_j - diag_rates

        # Termes de gain
        dndt = -total_loss

        # Redistribution des particules coagulées
        rate_flat = rate.view(ncells, -1)
        coag_map_3d = coag_map.flatten().expand(ncells, -1)

        dndt.scatter_add_(1, coag_map_3d, rate_flat)

        return dndt



    # ==================== INTÉGRATION PRINCIPALE ====================
    def computeIntegratedNs(self, grid, dt):
        """
        Intégration RK45 du système de suie

        Équation bilan:
        dns/dt = (nucléation) + (croissance) + (oxydation) + (coagulation)
        """
        K = self._compute_brownian_kernel_matrix(grid)  # (ncells, nclasses, nclasses)

        def rhs_soot(ns, t):
            """
            Right-hand side pour équations de population
            """
            ns = torch.clamp(ns, min=0.0)

            dn_nucleation = self.nucleation_LL(grid)
            dn_growth = self.growth_LL(ns, grid)
            dn_oxyd = self.oxidation_LL(ns, grid)
            dn_coag = self._coagulation_batched(ns, K, self.coag_map)

            return dn_nucleation + dn_growth + dn_oxyd + dn_coag

        # Intégration RK45
        integrator = RK45Integrator(
            rhs_soot,
            device=self.device,
            dtype=self.dtype,
        )

        ns0 = grid.cells.ns.clone()
        t_vals, ns_vals = integrator.integrate(ns0, t_span=(0.0, dt))
        ns_final = ns_vals[-1]

        ns_total = torch.clamp(ns_final, min=0.0)

        return ns_total

