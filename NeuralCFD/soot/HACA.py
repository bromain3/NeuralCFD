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

class HACA(DummySoot):
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




class HACA(DummySoot):
    def __init__(self, grid, thermo, C2H2name='C2H2', C2H2order=2.0, O2name='O2', OHname='OH', k_nuc=1.0e-15,
                 k_OH=1.0e4, k_O2=2.0e2,
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

        # self.coag_map = self.__build_coag_map(grid.nclasses, thermo.dtype, thermo.device)
        self.coag_map = self.__build_conservative_coag_map(grid)

        self.OHidx = thermo.thermoProp.species.index(OHname)
        self.O2idx = thermo.thermoProp.species.index(O2name)

    def __build_conservative_coag_map(self, grid):
        nclasses = grid.nclasses
        coag_map = torch.zeros((nclasses, nclasses), dtype=torch.long, device=grid.device)

        for i in range(nclasses):
            for j in range(nclasses):
                v_new = grid.meanParticuleVolume[i] + grid.meanParticuleVolume[j]
                k = torch.searchsorted(grid.classEdges, v_new) - 1
                k = torch.clamp(k, 0, nclasses - 1)
                coag_map[i, j] = k
        return coag_map

    def __compute_particle_radius(self, volume):
        """
        Calcule le rayon des particules à partir de leur volume
        Args:
            volume: Volume(s) des particules (m³) - peut être scalaire, array numpy ou tensor
        Returns:
            Rayon(s) en mètres (même type que l'entrée)
        """
        return (3 * volume / (4 * np.pi * RHO_SOOT)) ** (1 / 3)

    def __compute_particle_surface(self, volume):
        """
        Calcule la surface des particules à partir de leur volume
        Args:
            volume: Volume(s) des particules (m³) - peut être scalaire, array numpy ou tensor
        Returns:
            Surface(s) en m² (même type que l'entrée)
        """
        radius = self.__compute_particle_radius(volume)
        return 4 * np.pi * radius ** 2

    def get_particle_radii(self, grid):
        """
        Retourne les rayons moyens des particules pour chaque section
        Returns:
            Tensor de shape (nclasses,) contenant les rayons en mètres
        """
        volumes = grid.meanParticuleVolume
        if isinstance(volumes, torch.Tensor):
            volumes_np = volumes.cpu().numpy()
            radii_np = self.__compute_particle_radius(volumes_np)
            return torch.from_numpy(radii_np).to(volumes.device)
        else:
            return self.__compute_particle_radius(volumes)

    def get_particle_surfaces(self, grid):
        """
        Retourne les surfaces moyennes des particules pour chaque section
        Returns:
            Tensor de shape (nclasses,) contenant les surfaces en m²
        """
        volumes = grid.meanParticuleVolume
        if isinstance(volumes, torch.Tensor):
            volumes_np = volumes.cpu().numpy()
            surfaces_np = self.__compute_particle_surface(volumes_np)
            return torch.from_numpy(surfaces_np).to(volumes.device)
        else:
            return self.__compute_particle_surface(volumes)

    def nucleation(self, grid):
        C_PAH = grid.cells.Ys[:,
                self.C2H2idx] * grid.cells.rho  # TODO: optimize by using the precomputed value stored in grid.cells.Q
        R_nuc = self.k_nuc * C_PAH ** self.C2H2order

        dndt = torch.zeros_like(grid.cells.ns)
        dndt[:, 0] += R_nuc / grid.sectionSize[0]

        return dndt

    def __R_HACA(self, grid):
        T = grid.cells.T
        Ys = grid.cells.Ys
        k_HACA = self.A_HACA * T ** self.n_HACA * torch.exp(-self.Ea_HACA / (R * T))

        R_HACA = k_HACA * Ys[:, self.C2H2idx]

        return R_HACA * self.alpha

    def surfaceGrowth(self, ns, G, sectionSize):
        flux = G * ns  # flux en espace des tailles

        dndt = np.zeros_like(ns)
        dndt[1:] += flux[:-1] / sectionSize[1:]
        dndt[:-1] -= flux[:-1] / sectionSize[:-1]

        return dndt

    def oxydation(self, ns, L, sectionSize):

        flux = L * ns

        dndt = np.zeros_like(ns)
        dndt[:-1] += flux[1:] / sectionSize[:-1]
        dndt[1:] -= flux[1:] / sectionSize[1:]

        return dndt

    #TODO: ajouter la fragmentation ?
    def __cunningham_correction(self, radius, T):
        """Facteur de correction de Cunningham pour le régime de transition"""
        kn = lambda_g / radius
        cc = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
        return cc

    def __BrownianCoagulationKernel(self, grid, i, j):  # with correction for transitional regimes (so usable in continuum and transition regimes)
        """
        Noyau de coagulation Brownien avec correction de Cunningham
        pour le régime de transition
        """
        T = grid.cells.T.cpu().numpy()
        mu_gas = grid.cells.mu  # Viscosité dynamique (Pa.s)

        # Calcul des rayons à partir des volumes moyens
        vi = grid.meanParticuleVolume[i].cpu().numpy()
        vj = grid.meanParticuleVolume[j].cpu().numpy()

        ri = (3 * vi / (4 * np.pi * RHO_SOOT)) ** (1 / 3)
        rj = (3 * vj / (4 * np.pi * RHO_SOOT)) ** (1 / 3)

        # Corrections de Cunningham
        cc_i = self.__cunningham_correction(ri, T)
        cc_j = self.__cunningham_correction(rj, T)

        # Noyau de coagulation (régime de transition)
        beta = (2 * kB * T / (3 * mu_gas)) * (ri + rj) * (cc_i / ri + cc_j / rj)

        return beta

    def __compute_BrownianKernelMatrix(self, grid):
        """
        Calcule la matrice complète des noyaux de coagulation
        Shape: (ncells, nclasses, nclasses)
        """
        ncells = grid.cells.ns.shape[0]
        nclasses = grid.nclasses

        # Conversion en numpy pour les calculs
        T = grid.cells.T.cpu().numpy()
        mu_gas = grid.cells.mu.cpu().numpy() if hasattr(grid.cells, 'mu') else np.full(ncells, mu)

        v_nodes = grid.meanParticuleVolume.cpu().numpy()

        # Calcul des rayons
        radii = (3 * v_nodes / (4 * np.pi * RHO_SOOT)) ** (1 / 3)

        # Matrice des noyaux
        K = np.zeros((ncells, nclasses, nclasses))

        for cell_idx in range(ncells):
            T_cell = T[cell_idx]
            mu_cell = mu_gas[cell_idx]

            for i in range(nclasses):
                ri = radii[i]
                kn_i = lambda_g / ri
                cc_i = 1 + kn_i * (1.257 + 0.4 * np.exp(-1.1 / kn_i))

                for j in range(i, nclasses):
                    rj = radii[j]
                    kn_j = lambda_g / rj
                    cc_j = 1 + kn_j * (1.257 + 0.4 * np.exp(-1.1 / kn_j))

                    # Noyau de coagulation
                    beta = (2 * kB * T_cell / (3 * mu_cell)) * (ri + rj) * (cc_i / ri + cc_j / rj)

                    K[cell_idx, i, j] = beta
                    K[cell_idx, j, i] = beta  # Symétrie

        return torch.from_numpy(K).to(grid.device)

    def coagulation_dt_limit(self, grid, CFL=0.2):
        ns = grid.cells.ns  # (ncells, nclasses)
        K = self.__compute_BrownianKernelMatrix(grid)

        # perte : sum_j K_ij * n_j
        loss_rate = torch.einsum('cij,cj->ci', K, ns)

        # éviter division par zéro
        tau = torch.where(loss_rate > 0, 1.0 / loss_rate, torch.inf)

        dt_coag = CFL * torch.min(tau)
        return dt_coag

    def coagulation_numpy(self, ns, K_cell, coag_map, nclasses):
        """
        Calcul du terme de coagulation dN/dt pour une cellule
        Implémentation type Smoluchowski avec conservation de la masse
        """
        dNdt = np.zeros_like(ns)

        for i in range(nclasses):
            for j in range(i, nclasses):
                beta = K_cell[i, j]

                # Taux de coagulation
                if i == j:
                    rate = 0.5 * beta * ns[i] * ns[j]
                else:
                    rate = beta * ns[i] * ns[j]

                # Perte dans les sections i et j
                dNdt[i] -= rate
                if i != j:
                    dNdt[j] -= rate

                # Gain dans la section cible
                k = coag_map[i, j]
                dNdt[k] += rate

        return dNdt


    def computeIntegratedNs(self, grid, dt):
        """
        Intégration complète des termes sources/puits de densité numérique
        """
        # Étape 1 : Pré-calculs
        R_HACA = self.__R_HACA(grid)
        K = self.__compute_BrownianKernelMatrix(grid)

        # Calcul des surfaces des particules pour chaque section
        particle_surfaces = self.get_particle_surfaces(grid)  # (nclasses,)

        # Taux de croissance surfacique
        G = (particle_surfaces[None, :] * R_HACA[:, None]).cpu().numpy()

        # Taux d'oxydation
        L = (particle_surfaces[None, :] * (
                                                  self.k_OH * grid.cells.Ys[:, self.OHidx] * grid.cells.rho +
                                                  self.k_O2 * grid.cells.Ys[:, self.O2idx] * grid.cells.rho
                                          )[:, None]).cpu().numpy()

        # Conversion pour numpy
        sectionSize = grid.sectionSize.cpu().numpy()
        K_numpy = K.cpu().numpy()
        coag_map = self.coag_map.cpu().numpy()

        # Étape 2 : Nucléation (terme indépendant)
        dn_nucleation = self.nucleation(grid) * dt

        # Étape 3 : Intégration des termes couplés (cellule par cellule)
        ncells = grid.cells.ns.shape[0]
        nclasses = grid.nclasses

        ns0_np = grid.cells.ns.cpu().numpy()
        n_total_np = np.zeros_like(ns0_np)

        for cell_idx in range(ncells):
            ns0 = ns0_np[cell_idx, :]

            # Paramètres spécifiques à la cellule
            G_cell = G[cell_idx, :]
            L_cell = L[cell_idx, :]
            K_cell = K_numpy[cell_idx, :, :]

            # Fonction RHS pour solve_ivp
            def rhs(t, ns):
                ns = np.maximum(ns, 0.0)  # Assurer positivité

                dn_growth = self.surfaceGrowth(ns, G_cell, sectionSize)
                dn_oxyd = self.oxydation(ns, L_cell, sectionSize)
                dn_coag = self.coagulation_numpy(ns, K_cell, coag_map, nclasses)

                return dn_growth + dn_oxyd + dn_coag

            # Résolution ODE
            sol = solve_ivp(
                rhs,
                [0, dt],
                ns0,
                method='RK45',  # Méthode pour systèmes raides
                atol=1e-10,
                rtol=1e-8
            )

            # Solution finale
            ns_final = np.maximum(sol.y[:, -1], 0.0)
            n_total_np[cell_idx, :] = ns_final

        # Conversion en torch et ajout de la nucléation
        n_total = torch.from_numpy(n_total_np).to(grid.cells.ns.device)
        n_total += dn_nucleation

        # Affichage du pas de temps limite
        print(f"Coagulation dt limit: {self.coagulation_dt_limit(grid):.2e} s")

        return n_total