from NeuralCFD.soot.DummySoot import DummySoot
import NeuralCFD.utilities_functions.primitive_converter as primitive_converter

import torch
import numpy as np


import torch
import numpy as np
from scipy.integrate import solve_ivp
from NeuralCFD.soot.DummySoot import DummySoot

R = 8.31446261815324
kB = 1.380649e-23
mu = 2.0e-5
RHO_SOOT = 1800  # Densité des suies (kg/m³)
lambda_g = 400e-9  # Libre parcours moyen du gaz (m)


"""
R = 8.31446261815324
kB = 1.380649e-23
mu = 2.0e-5

class oneWayHACA(DummySoot):
    #TODO: make the two way coupling (consumption of the PAH used for nucleation)
    def __init__(self, grid, thermo, C2H2name='C2H2', C2H2order=2.0, O2name='O2', OHname='OH', k_nuc=1.0e-15, k_OH=1.0e4, k_O2=2.0e2,
                 A_HACA=1.0e5, n_HACA=0.0, Ea_HACA=1.5e5, alpha=1.0):
        self.C2H2idx = thermo.thermoProp.species.index(C2H2name)
        self.k_nuc = k_nuc  # 1.0e-15
        self.C2H2order = C2H2order  # 2.0
        self.k_OH = k_OH  # 1.0e4
        self.k_O2 = k_O2  # 2.0e2

        # the specific value of the following parameters depend on the temperature at which the model must be accurate
        self.A_HACA = A_HACA  # 1.0e5
        self.n_HACA = n_HACA  # 0.0
        self.Ea_HACA = Ea_HACA  # 1.5e5
        self.alpha = alpha  # 1.0 (between 0.5 and 3.0)

        #self.coag_map = self.__build_coag_map(grid.nclasses, thermo.dtype, thermo.device)
        self.coag_map = self.__build_conservative_coag_map(grid)

        self.OHidx = thermo.thermoProp.species.index(OHname)
        self.O2idx = thermo.thermoProp.species.index(O2name)

    def __build_coag_map(self, nclasses, dtype, device):
        coag_map = torch.full((nclasses, nclasses), -1, dtype=torch.long, device=device)
        for i in range(nclasses):
            for j in range(nclasses):
                coag_map[i, j] = min(i + j, nclasses - 1)
        return coag_map

    def __build_conservative_coag_map(self, grid):

        #meanParticuleVolume : (nclasses,)
        #v_edges             : (nclasses+1,)
        
        nclasses = grid.nclasses
        coag_map = torch.zeros((nclasses, nclasses), dtype=torch.long, device=grid.device)

        for i in range(nclasses):
            for j in range(nclasses):
                v_new = grid.meanParticuleVolume[i] + grid.meanParticuleVolume[j]

                # recherche de la section cible
                k = torch.searchsorted(grid.classEdges, v_new) - 1

                # saturation sur la dernière classe
                k = torch.clamp(k, 0, nclasses - 1)

                coag_map[i, j] = k

        return coag_map

    def nucleation(self, grid):
        C_PAH = grid.cells.Ys[:, self.C2H2idx] * grid.cells.rho  # TODO: optimize by using the precomputed value stored in grid.cells.Q
        R_nuc = self.k_nuc * C_PAH ** self.C2H2order

        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, 0] += R_nuc / grid.sectionSize[0]

        return dndt

    def __R_HACA(self, grid):
        T = grid.cells.T
        Ys = grid.cells.Ys
        k_HACA = self.A_HACA * T ** self.n_HACA * torch.exp(-self.Ea_HACA/(R * T))

        R_HACA = k_HACA * Ys[:, self.C2H2idx]

        return R_HACA * self.alpha

    def surfaceGrowth(self, grid):
        G = grid.paticleSurface * self.__R_HACA(grid)[:, None]  # shape (ncells, nclasses)

        flux = G * grid.cells.ns  # flux en espace des tailles

        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, 1:] += flux[:, :-1] / grid.sectionSize[1:]
        dndt[:, :-1] -= flux[:, :-1] / grid.sectionSize[:-1]

        return dndt

    def oxydation(self, grid):
        L = grid.paticleSurface * (self.k_OH * grid.cells.Ys[:, self.OHidx] * grid.cells.rho + self.k_O2 * grid.cells.Ys[:, self.O2idx] * grid.cells.rho)[:, None]

        flux = L * grid.cells.ns

        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, :-1] += flux[:, 1:] / grid.sectionSize[:-1]
        dndt[:, 1:] -= flux[:, 1:] / grid.sectionSize[1:]

        return dndt

    def __BrownianCoagulationKernel(self, grid, i, j):  # OK for continuum regimes but not for transitional regimes
        T = grid.cells.T  # (ncells,)
        di = grid.particuleDiameter[i]
        dj = grid.particuleDiameter[j]

        K_ij = (2 * kB * T / (3 * mu)) * (1 / di + 1 / dj) * (di + dj)
        return K_ij  # (ncells,)

    def __compute_BrownianKernelMatrix(self, grid):
        T = grid.cells.T[:, None, None]  # (ncells,1,1)
        d = grid.particuleDiameter  # (nclasses,)

        di = d.view(1, -1, 1)
        dj = d.view(1, 1, -1)

        K = (2 * kB * T / (3 * mu)) * (1 / di + 1 / dj) * (di + dj)
        return K  # (ncells, nclasses, nclasses)

    def coagulation_dt_limit(self, grid, CFL=0.2):
        ns = grid.cells.ns  # (ncells, nclasses)
        K = self.__compute_BrownianKernelMatrix(grid)

        # perte : sum_j K_ij * n_j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)

        # éviter division par zéro
        tau = torch.where(loss_rate > 0, 1.0 / loss_rate, torch.inf)

        dt_coag = CFL * torch.min(tau)
        return dt_coag

    def coagulation_sequential(self, grid):
        ns = grid.cells.ns
        dndt = torch.zeros_like(ns)

        for i in range(grid.nclasses):
            for j in range(grid.nclasses):
                k = self.coag_map[i, j]
                K_ij = self.__BrownianCoagulationKernel(grid, i, j)
                dndt[:, k] += 0.5 * K_ij * ns[:, i] * ns[:, j]

        # perte
        loss = torch.zeros_like(ns)
        for i in range(grid.nclasses):
            for j in range(grid.nclasses):
                K_ij = self.__BrownianCoagulationKernel(grid, i, j)
                loss[:, i] += K_ij * ns[:, j]

        dndt -= ns * loss

        return dndt

    def coagulation(self, grid):
        ns = grid.cells.ns  # (ncells, nclasses)
        ncells, nclasses = ns.shape

        K = self.__compute_BrownianKernelMatrix(grid)
        # n_i * n_j
        nij = ns[:, :, None] * ns[:, None, :]  # (ncells, nclasses, nclasses)
        # taux de coagulation
        R_ij = 0.5 * K * nij  # facteur 1/2 pour symétrie
        # projection vers k
        dndt_gain = torch.zeros_like(ns)
        # flatten (i,j) → k
        k_map = self.coag_map.view(-1)  # (nclasses*nclasses,)
        R_flat = R_ij.view(ncells, -1)  # (ncells, nclasses*nclasses)
        # accumulation
        dndt_gain.scatter_add_(
            dim=1,
            index=k_map.unsqueeze(0).expand(ncells, -1),
            src=R_flat
        )
        # sum_j K_ij * n_j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)

        dndt_loss = ns * loss_rate

        dndt = dndt_gain - dndt_loss
        return dndt

    def computeSources(self, grid, dt):
        dndt = self.nucleation(grid) + self.surfaceGrowth(grid) + self.oxydation(grid) + self.coagulation_sequential(grid)

        return dndt
    # TODO: make the two way coupling (consumption of the PAH used for nucleation). It will have an impact on numerical integration
    """


class oneWayHACA_CPU(DummySoot):  # basé sur la concentration de C2H2, pas vraiment de PAH
    def __init__(self, grid, thermo, C2H2name='C2H2', C2H2order=1.0, O2name='O2', OHname='OH',
                 A_nuc=1.0e5, EA_nuc=180e3, k_OH=1.0e4, k_O2=2.0e2,
                 A_HACA=1.0e5, n_HACA=0.0, Ea_HACA=1.5e5, alpha=1.0):
        # Indices d'espèces
        self.C2H2idx = thermo.thermoProp.species.index(C2H2name)
        self.OHidx = thermo.thermoProp.species.index(OHname)
        self.O2idx = thermo.thermoProp.species.index(O2name)

        # Constantes cinétiques
        self.A_nuc = A_nuc
        self.EA_nuc = EA_nuc
        self.C2H2order = C2H2order
        self.k_OH = k_OH
        self.k_O2 = k_O2
        self.A_HACA = A_HACA
        self.n_HACA = n_HACA
        self.Ea_HACA = Ea_HACA
        self.alpha = alpha

        # Coagulation map
        self.coag_map = self._build_conservative_coag_map(grid)

        # Pré-calculs géométriques (invariants dans le temps)
        self._precompute_geometry(grid)

    def _build_conservative_coag_map(self, grid):
        """Construction vectorisée de la coagulation map"""
        nclasses = grid.nclasses
        v = grid.meanParticuleVolume  # (nclasses,)

        # Broadcasting : v_new[i,j] = v[i] + v[j]
        v_new = v[:, None] + v[None, :]  # (nclasses, nclasses)

        # Recherche vectorisée
        k = torch.searchsorted(grid.classEdges, v_new.flatten()) - 1
        k = torch.clamp(k, 0, nclasses - 1).reshape(nclasses, nclasses)

        return k

    def _precompute_geometry(self, grid):
        """Pré-calcul des propriétés géométriques (rayons, surfaces)"""
        v = grid.meanParticuleVolume.cpu().numpy()

        # Rayons et surfaces (numpy pour compatibilité)
        self.radii = (3 * v / (4 * np.pi * RHO_SOOT)) ** (1 / 3)  # (nclasses,)
        self.surfaces = 4 * np.pi * self.radii ** 2  # (nclasses,)

        # Corrections de Cunningham (indépendantes de T au premier ordre)
        kn = lambda_g / self.radii
        self.cc = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))  # (nclasses,)

        # Tenseur pour calculs torch
        self.surfaces_torch = torch.from_numpy(self.surfaces).to(grid.device)

    def nucleation(self, grid):
        """Nucléation - terme source dans la première section"""
        C_C2H2 = grid.cells.Ys[:, self.C2H2idx] * grid.cells.rho
        R_nuc = self.A_nuc * torch.exp(-self.EA_nuc / (R * grid.cells.T)) * C_C2H2 ** self.C2H2order

        v_nuc = grid.meanParticuleVolume[0]  # soot created in the smallest section
        m_p = RHO_SOOT * v_nuc

        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, 0] = R_nuc / m_p

        return dndt

    def _compute_R_HACA(self, grid):
        """Taux de croissance oneWayHACA"""
        T = grid.cells.T
        k_HACA = self.A_HACA * T ** self.n_HACA * torch.exp(-self.Ea_HACA / (R * T))
        return k_HACA * grid.cells.Ys[:, self.C2H2idx] * self.alpha

    def _compute_brownian_kernel_matrix(self, grid):
        """
        Calcul optimisé de la matrice des noyaux de coagulation
        Utilise broadcasting pour éviter les boucles explicites
        """
        ncells = grid.cells.ns.shape[0]
        nclasses = grid.nclasses

        T = grid.cells.T.cpu().numpy()  # (ncells,)
        mu_gas = grid.cells.mu.cpu().numpy() if hasattr(grid.cells, 'mu') else np.full(ncells, mu)  # (ncells,)  (using np.full(ncells, mu) is as efficient or even a bit more efficient than using mu directly)

        # Broadcasting : (ncells, 1, 1) × (1, nclasses, nclasses)
        ri = self.radii[None, :, None]  # (1, nclasses, 1)
        rj = self.radii[None, None, :]  # (1, 1, nclasses)
        cc_i = self.cc[None, :, None]
        cc_j = self.cc[None, None, :]

        # Constante : 2*kB*T / (3*mu)
        prefactor = (2 * kB * T / (3 * mu_gas))[:, None, None]  # (ncells, 1, 1)

        # Noyau de coagulation (vectorisé)
        K = prefactor * (ri + rj) * (cc_i / ri + cc_j / rj)  # (ncells, nclasses, nclasses)

        return torch.from_numpy(K).to(grid.device)

    def coagulation_dt_limit(self, grid, CFL=0.2):
        """Estimation du pas de temps limite pour la coagulation"""
        ns = grid.cells.ns
        K = self._compute_brownian_kernel_matrix(grid)

        # Taux de perte : sum_j K_ij * n_j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)

        # Temps caractéristique (éviter /0)
        tau = torch.where(loss_rate > 1e-30, 1.0 / loss_rate, 1e10)

        return CFL * torch.min(tau)

    @staticmethod
    def _surface_growth(ns, G, sectionSize):
        """Croissance surfacique - flux en espace des tailles"""
        flux = G * ns

        dndt = np.zeros_like(ns)
        dndt[1:] += flux[:-1] / sectionSize[1:]
        dndt[:-1] -= flux[:-1] / sectionSize[:-1]

        return dndt

    @staticmethod
    def _oxidation(ns, L, sectionSize):
        """Oxydation - flux inverse en espace des tailles"""
        flux = L * ns

        dndt = np.zeros_like(ns)
        dndt[:-1] += flux[1:] / sectionSize[:-1]
        dndt[1:] -= flux[1:] / sectionSize[1:]

        return dndt

    @staticmethod
    def _coagulation(ns, K_cell, coag_map, nclasses):
        """
        Coagulation Brownienne optimisée
        Utilise la symétrie et évite les calculs redondants
        """
        dNdt = np.zeros_like(ns)

        # Boucle optimisée avec symétrie
        for i in range(nclasses):
            ni = ns[i]
            if ni < 1e-30:  # Skip cellules vides
                continue

            for j in range(i, nclasses):
                nj = ns[j]
                if nj < 1e-30:
                    continue

                beta = K_cell[i, j]

                # Taux de coagulation
                rate = (0.5 if i == j else 1.0) * beta * ni * nj

                # Pertes
                dNdt[i] -= rate
                if i != j:
                    dNdt[j] -= rate

                # Gain dans section cible
                dNdt[coag_map[i, j]] += rate

        return dNdt

    def computeIntegratedNs(self, grid, dt):
        """
        Intégration optimisée des équations de population
        """
        # === Pré-calculs globaux (vectorisés) ===
        R_HACA = self._compute_R_HACA(grid)  # (ncells,)
        K = self._compute_brownian_kernel_matrix(grid)  # (ncells, nclasses, nclasses)


        # TODO: check if the growing rates are expressed in particles per m3 per s and if the k should not be computed as arhenius
        # Taux de croissance surfacique G[cell, class]
        G = (self.surfaces_torch[None, :] * R_HACA[:, None]).cpu().numpy()

        # Taux d'oxydation L[cell, class]
        oxyd_rate = (self.k_OH * grid.cells.Ys[:, self.OHidx] +
                     self.k_O2 * grid.cells.Ys[:, self.O2idx]) * grid.cells.rho
        L = (self.surfaces_torch[None, :] * oxyd_rate[:, None]).cpu().numpy()

        # Conversion numpy
        sectionSize = grid.sectionSize.cpu().numpy()
        K_numpy = K.cpu().numpy()
        coag_map = self.coag_map.cpu().numpy()
        ns0_all = grid.cells.ns.cpu().numpy()

        # Nucléation (terme indépendant)
        dn_nucleation = self.nucleation(grid) * dt

        # === Intégration cellule par cellule ===
        ncells, nclasses = ns0_all.shape
        ns_final_all = np.empty_like(ns0_all)

        for cell_idx in range(ncells):
            ns0 = ns0_all[cell_idx]

            # Skip cellules vides
            if np.sum(ns0) < 1e-30:
                ns_final_all[cell_idx] = ns0
                continue

            # Paramètres locaux
            G_cell = G[cell_idx]
            L_cell = L[cell_idx]
            K_cell = K_numpy[cell_idx]

            # Fonction RHS optimisée
            def rhs(t, ns):
                ns = np.maximum(ns, 0.0)

                dn_growth = self._surface_growth(ns, G_cell, sectionSize)
                dn_oxyd = self._oxidation(ns, L_cell, sectionSize)
                dn_coag = self._coagulation(ns, K_cell, coag_map, nclasses)

                return dn_growth + dn_oxyd + dn_coag


            sol = solve_ivp(
                rhs, [0, dt], ns0,
                method='RK45',  # BDF Meilleur pour systèmes raides
            )
            ns_final_all[cell_idx] = sol.y[:, -1]


        # Conversion torch et ajout nucléation
        ns_total = np.maximum(torch.from_numpy(ns_final_all).to(grid.device) + dn_nucleation, 0.0)

        # Diagnostic
        #dt_limit = self.coagulation_dt_limit(grid)
        #print(f"Coag dt limit: {dt_limit:.2e} s | dt/dt_limit: {dt / dt_limit:.2f}")

        return ns_total