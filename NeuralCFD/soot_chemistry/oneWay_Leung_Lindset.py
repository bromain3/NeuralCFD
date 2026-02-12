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

        self._precompute_scatter_indices(grid.ncells, grid.nclasses, self.coag_map)


        # Précalcul du masque triangulaire
        nclasses = grid.nclasses
        triu_mask = torch.triu(torch.ones(nclasses, nclasses, dtype=self.dtype, device=self.device))
        diag_mask = torch.eye(nclasses, dtype=self.dtype, device=self.device)
        self._triu_mask_coag = triu_mask - 0.5 * diag_mask

        # ============== GRID INFO ==============
        self.nclasses = grid.nclasses

    # ==================== COAGULATION MAP (REFACTORISÉE) ====================
    def _build_conservative_coag_map(self, grid):
        """
        Construction de la carte de coagulation - VERSION COHÉRENTE avec basic_loops

        LOGIQUE (identique à _coagulation_basic_loops):
        Pour chaque paire (j, k):
        1. v_new = v_j + v_k
        2. Trouver k_low tel que: v[k_low] <= v_new < v[k_low+1]
        3. k_high = k_low + 1 (sauf si overflow → k_high = k_low)
        4. frac_low = (v_high - v_new) / (v_high - v_low)

        Cette carte sera utilisée par _coagulation_vectorized
        """
        nclasses = grid.nclasses
        v = self.m_class / self.RHO_SOOT  # (nclasses,) volumes en m³

        # Matrice des volumes résultants v_new[j,k] = v[j] + v[k]
        v_j = v[:, None]  # (nclasses, 1)
        v_k = v[None, :]  # (1, nclasses)
        v_new = v_j + v_k  # (nclasses, nclasses)

        # Initialisation des tenseurs de sortie
        k_low = torch.zeros((nclasses, nclasses), dtype=torch.long, device=self.device)
        k_high = torch.zeros((nclasses, nclasses), dtype=torch.long, device=self.device)
        frac_low = torch.zeros((nclasses, nclasses), dtype=self.dtype, device=self.device)

        # Remplissage - même algorithme que dans basic_loops
        for j in range(nclasses):
            for k in range(nclasses):
                v_target = v_new[j, k].item()

                # Chercher k_target_low tel que v[k_target_low] <= v_target < v[k_target_low+1]
                k_target_low = nclasses - 1  # Valeur par défaut (overflow)

                for i in range(nclasses - 1):
                    if v[i] <= v_target < v[i + 1]:
                        k_target_low = i
                        break

                k_target_high = min(k_target_low + 1, nclasses - 1)

                # Calcul de la fraction (interpolation linéaire)
                if k_target_low == k_target_high:
                    # Une seule classe de destination
                    frac = 1.0
                else:
                    # Deux classes: distribution proportionnelle
                    v_low = v[k_target_low].item()
                    v_high = v[k_target_high].item()

                    if abs(v_high - v_low) > 1e-30:
                        frac = (v_high - v_target) / (v_high - v_low)
                        frac = max(0.0, min(1.0, frac))
                    else:
                        frac = 1.0

                # Stockage
                k_low[j, k] = k_target_low
                k_high[j, k] = k_target_high
                frac_low[j, k] = frac

        # Validation
        assert torch.all(k_low >= 0) and torch.all(k_low < nclasses), "k_low hors limites"
        assert torch.all(k_high >= 0) and torch.all(k_high < nclasses), "k_high hors limites"
        assert torch.all(frac_low >= -1e-6), f"frac_low < 0: min={frac_low.min()}"
        assert torch.all(frac_low <= 1.0 + 1e-6), f"frac_low > 1: max={frac_low.max()}"

        return {
            'k_low': k_low,
            'k_high': k_high,
            'frac_low': torch.clamp(frac_low, 0.0, 1.0),
            'v_new': v_new
        }

    def _precompute_scatter_indices(self, ncells, nclasses, coag_map):
        """Précalcul des indices pour scatter_add (version ultra)"""
        k_low = coag_map['k_low']
        k_high = coag_map['k_high']
        frac_low = coag_map['frac_low']

        cell_idx = torch.arange(ncells, device=self.device)[:, None, None]
        cell_idx = cell_idx.expand(ncells, nclasses, nclasses)

        k_low_exp = k_low[None, :, :].expand(ncells, nclasses, nclasses)
        k_high_exp = k_high[None, :, :].expand(ncells, nclasses, nclasses)

        flat_idx_low = (cell_idx * nclasses + k_low_exp).flatten()
        flat_idx_high = (cell_idx * nclasses + k_high_exp).flatten()

        frac_low_exp = frac_low[None, :, :].expand(ncells, nclasses, nclasses)
        frac_low_flat = frac_low_exp.flatten()

        self._flat_idx_low = flat_idx_low
        self._flat_idx_high = flat_idx_high
        self._frac_low_flat = frac_low_flat

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

    def _coagulation_semi_vectorized(self, ns, K, coag_map):
        """
        Version vectorisée utilisant la coag_map

        STRUCTURE:
        1. PERTE: -n_i × Σ_j K_{ij} × n_j
        2. GAIN: distribution des collisions via coag_map

        Cette version donne les MÊMES résultats que basic_loops
        mais est plus rapide (moins de boucles FOR)
        """
        ncells, nclasses = ns.shape
        dndt = torch.zeros_like(ns)

        # ========== TERME DE PERTE ==========
        # Pour chaque classe i: n_i disparaît en collisant avec toutes les j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)  # (ncells, nclasses)
        dndt -= ns * loss_rate

        # ========== TERME DE GAIN ==========
        # Taux de collision pour toutes les paires (j,k)
        n_j = ns[:, :, None]  # (ncells, nclasses, 1)
        n_k = ns[:, None, :]  # (ncells, 1, nclasses)
        collision_rate = K * n_j * n_k  # (ncells, nclasses, nclasses)

        # Facteur 1/2 pour auto-coagulation (diagonale j=k)
        diag_mask = torch.eye(nclasses, device=self.device, dtype=self.dtype)[None, :, :]
        collision_rate = collision_rate * (1.0 - 0.5 * diag_mask)

        # Récupération des indices de destination depuis coag_map
        k_low = coag_map['k_low']  # (nclasses, nclasses)
        k_high = coag_map['k_high']  # (nclasses, nclasses)
        frac_low = coag_map['frac_low']  # (nclasses, nclasses)

        # Distribution du gain dans les classes de destination
        # Boucle FOR simplifiée (peut être optimisée avec scatter_add si besoin)
        for j in range(nclasses):
            for k in range(j, nclasses):  # k >= j pour éviter double comptage
                i_low = k_low[j, k].item()
                i_high = k_high[j, k].item()
                frac = frac_low[j, k].item()

                rate = collision_rate[:, j, k]  # (ncells,)

                # Ajouter aux classes de destination
                dndt[:, i_low] += frac * rate
                if i_high != i_low:
                    dndt[:, i_high] += (1.0 - frac) * rate

        return dndt

    # ==================== COAGULATION (BASIC LOOPS - RÉFÉRENCE) ====================
    def _coagulation_basic_loops(self, ns, K, coag_map):
        """
        Version avec boucles FOR - RÉFÉRENCE pour validation
        Garde cette méthode pour debug!
        """
        ncells, nclasses = ns.shape
        dndt = torch.zeros_like(ns)

        v = self.m_class / self.RHO_SOOT

        for icell in range(ncells):
            n = ns[icell, :]
            K_cell = K[icell, :, :]

            # PERTE
            for i in range(nclasses):
                loss_i = 0.0
                for j in range(nclasses):
                    loss_i += K_cell[i, j] * n[j]
                dndt[icell, i] -= n[i] * loss_i

            # GAIN
            for j in range(nclasses):
                for k in range(j, nclasses):
                    v_new = v[j] + v[k]

                    k_target_low = 0
                    for i in range(nclasses - 1):
                        if v[i] <= v_new < v[i + 1]:
                            k_target_low = i
                            break
                    else:
                        k_target_low = nclasses - 1

                    k_target_high = min(k_target_low + 1, nclasses - 1)

                    if k_target_low == k_target_high:
                        frac_low = 1.0
                    else:
                        v_low = v[k_target_low]
                        v_high = v[k_target_high]
                        frac_low = (v_high - v_new) / (v_high - v_low)
                        frac_low = max(0.0, min(1.0, frac_low))

                    if j == k:
                        rate = 0.5 * K_cell[j, k] * n[j] * n[k]
                    else:
                        rate = K_cell[j, k] * n[j] * n[k]

                    dndt[icell, k_target_low] += frac_low * rate
                    if k_target_high != k_target_low:
                        dndt[icell, k_target_high] += (1.0 - frac_low) * rate

        return dndt


    def _coagulation_ultra_optimized(self, ns, K, coag_map):
        """
        Version ULTRA optimisée – conservative
        Aucun for-loop, aucun recalcul d'indices
        """
        ncells, nclasses = ns.shape

        # =========================
        # COLLISION UNIQUE j <= k
        # =========================
        n_j = ns[:, :, None]
        n_k = ns[:, None, :]

        collision_rate = K * n_j * n_k
        collision_rate = collision_rate * self._triu_mask_coag[None, :, :]

        # =========================
        # PERTE
        # =========================
        loss_j = torch.sum(collision_rate, dim=2)
        loss_k = torch.sum(collision_rate, dim=1)

        dndt = -loss_j - loss_k

        # =========================
        # GAIN (scatter)
        # =========================
        collision_flat = collision_rate.flatten()

        values_low = collision_flat * self._frac_low_flat
        values_high = collision_flat * (1.0 - self._frac_low_flat)

        dndt_flat = dndt.reshape(-1)
        dndt_flat.scatter_add_(0, self._flat_idx_low, values_low)
        dndt_flat.scatter_add_(0, self._flat_idx_high, values_high)

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
            dn_coag = self._coagulation_ultra_optimized(ns, K, self.coag_map)

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

