from abc import ABC, abstractmethod
import torch
import math


class DummyCoagulation_vect(ABC):
    """
    Classe de base pour les modèles de coagulation, implémentant la méthode SECT
    (Sectional) de sootlib avec splitting pour conservation de la masse et du nombre.
    VERSION VECTORISÉE - tous les précalculs géométriques sont faits dans __init__
    """

    def __init__(self, grid, RHO_SOOT):
        """
        Args:
            grid: objet contenant les informations sur le maillage de tailles
            RHO_SOOT: densité de la suie (kg/m³)
        """
        self.kB = torch.tensor(1.380649e-23, device=grid.device, dtype=grid.dtype)  # J/K
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)
        self.binGrowthFactor = grid.binGrowthFactor

        # Précalcul de 1/ln(F) pour localiser les bins cibles
        self.ilnF = 1.0 / math.log(self.binGrowthFactor)

        # Masse d'une particule dans chaque section (kg)
        self.mBins = grid.meanParticuleVolume * self.RHO_SOOT  # (nclasses,)
        self.nclasses = len(self.mBins)
        self.ncells = grid.ncells
        self.device = grid.device
        self.dtype = grid.dtype

        # Précalcul du masque triangulaire pour éviter double comptage
        # mask[i,j] = 1 si i < j, 0.5 si i == j, 0 sinon
        triu_mask = torch.triu(torch.ones(self.nclasses, self.nclasses,
                                          dtype=self.dtype, device=self.device))
        diag_mask = torch.eye(self.nclasses, dtype=self.dtype, device=self.device)
        self._triu_mask = triu_mask - 0.5 * diag_mask  # (nclasses, nclasses)

        # Précalcul de la carte de coagulation (splitting conservatif)
        self._precompute_coagulation_map()

        # Précalcul des indices pour scatter_add
        self._precompute_scatter_indices()

    def _precompute_coagulation_map(self):
        """
        Précalcul de la carte de splitting pour toutes les paires (i,j).
        Pour chaque collision i+j, on détermine:
        - k_low, k_high: les bins cibles pour le splitting
        - frac_low: fraction de particules dans k_low

        Formule de splitting (conservation masse + nombre):
        m_combined = mBins[i] + mBins[j]
        k = log(m_combined / mBins[0]) / log(F)

        Si k < nclasses-1: splitting entre floor(k) et ceil(k)
        Xk = (m[k+1] - m_combined) / (m[k+1] - m[k])
        """
        nclasses = self.nclasses

        # Matrices des masses combinées m[i] + m[j]
        mi = self.mBins[:, None]  # (nclasses, 1)
        mj = self.mBins[None, :]  # (1, nclasses)
        m_combined = mi + mj  # (nclasses, nclasses)

        # Calcul de k_float = log(m_combined / m0) / log(F)
        k_float = self.ilnF * torch.log(m_combined / self.mBins[0])

        # floor(k) et ceil(k), clampés à [0, nclasses-1]
        k_low = torch.clamp(torch.floor(k_float).long(), 0, nclasses - 1)
        k_high = torch.clamp(k_low + 1, 0, nclasses - 1)

        # Initialisation des fractions
        frac_low = torch.zeros((nclasses, nclasses), dtype=self.dtype, device=self.device)

        # Masque pour les cas où k_high != k_low (splitting requis)
        split_mask = (k_low < nclasses - 1)

        # Cas sans splitting: tout va dans k_low
        # Conservation de la masse: frac = m_combined / m[k]
        no_split_mask = ~split_mask
        if no_split_mask.any():
            k_nosplit = k_low[no_split_mask]
            m_combined_nosplit = m_combined[no_split_mask]
            m_k_nosplit = self.mBins[k_nosplit]
            # Pour conserver la masse dans la dernière bin
            frac_low[no_split_mask] = m_combined_nosplit / m_k_nosplit

        # Cas avec splitting: interpolation linéaire
        if split_mask.any():
            k_l = k_low[split_mask]
            k_h = k_high[split_mask]
            m_l = self.mBins[k_l]
            m_h = self.mBins[k_h]
            m_comb = m_combined[split_mask]

            # Xk = (m[k+1] - m_combined) / (m[k+1] - m[k])
            frac_low[split_mask] = (m_h - m_comb) / (m_h - m_l)

        # Clamping pour sécurité numérique
        frac_low = torch.clamp(frac_low, 0.0, 1.0)

        # Stockage
        self._coag_map = {
            'k_low': k_low,  # (nclasses, nclasses)
            'k_high': k_high,  # (nclasses, nclasses)
            'frac_low': frac_low  # (nclasses, nclasses)
        }

    def _precompute_scatter_indices(self):
        """
        Précalcul des indices pour scatter_add.
        Transforme les indices (icell, i, j) -> (icell, k_target) en indices plats.
        """
        k_low = self._coag_map['k_low']
        k_high = self._coag_map['k_high']
        frac_low = self._coag_map['frac_low']

        # Expansion pour toutes les cellules
        # cell_idx: (ncells, 1, 1) -> (ncells, nclasses, nclasses)
        cell_idx = torch.arange(self.ncells, device=self.device, dtype=torch.int64)
        cell_idx = cell_idx[:, None, None].expand(self.ncells, self.nclasses, self.nclasses)

        # k_low, k_high: (1, nclasses, nclasses) -> (ncells, nclasses, nclasses)
        k_low_exp = k_low[None, :, :].expand(self.ncells, self.nclasses, self.nclasses)
        k_high_exp = k_high[None, :, :].expand(self.ncells, self.nclasses, self.nclasses)

        # Indices plats: icell * nclasses + k_target
        flat_idx_low = (cell_idx * self.nclasses + k_low_exp).flatten()
        flat_idx_high = (cell_idx * self.nclasses + k_high_exp).flatten()

        # Fractions pour scatter_add
        frac_low_exp = frac_low[None, :, :].expand(self.ncells, self.nclasses, self.nclasses)
        frac_low_flat = frac_low_exp.flatten()

        # Stockage
        self._flat_idx_low = flat_idx_low
        self._flat_idx_high = flat_idx_high
        self._frac_low_flat = frac_low_flat


    @abstractmethod
    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcule la matrice du noyau de coagulation K[i,j]

        Args:
            ns: concentrations en nombre (ncells, nclasses)
            ys: fractions massiques (ncells, nspecies)
            rhos: densités (ncells,)
            Ts: températures (ncells,)

        Returns:
            K: matrice de noyau de coagulation (ncells, nclasses, nclasses)
               K[icell, i, j] = taux de coagulation entre bins i et j (#/m³/s)
        """
        raise NotImplementedError('compute_kernel_matrix must be implemented in child class')

    def dndt(self, ns, ys, rhos, Ts):
        """
        Calcule le terme source de coagulation - VERSION VECTORISÉE

        Algorithme:
        1. Calcul du noyau K[icell, i, j]
        2. PERTE: pour chaque bin i, somme sur j de K[i,j] * n[i] * n[j]
        3. GAIN: pour chaque collision (i,j), splitting conservatif vers bins cibles

        Utilise le masque triangulaire pour éviter double comptage:
        - Si i < j: taux = K[i,j] * n[i] * n[j]
        - Si i == j: taux = 0.5 * K[i,i] * n[i]²

        Args:
            ns: concentrations en nombre (ncells, nclasses) en #/m³
            ys: fractions massiques (ncells, nspecies)
            rhos: densités (ncells,)
            Ts: températures (ncells,)

        Returns:
            Scoa: termes source de coagulation (ncells, nclasses) en #/m³/s
        """
        # Calcul du noyau de coagulation
        K = self.compute_kernel_matrix(ns, ys, rhos, Ts)  # (ncells, nclasses, nclasses)

        # ========== COLLISIONS (avec masque triangulaire) ==========
        # n[i] * n[j] pour toutes les paires
        n_i = ns[:, :, None]  # (ncells, nclasses, 1)
        n_j = ns[:, None, :]  # (ncells, 1, nclasses)

        # Taux de collision: K[i,j] * n[i] * n[j] * mask
        # mask = 1 si i < j, 0.5 si i == j (pour éviter double comptage)
        collision_rate = K * n_i * n_j * self._triu_mask[None, :, :]  # (ncells, nclasses, nclasses)

        # ========== PERTE ==========
        # Chaque bin i perd des particules par collision avec tous les j
        # Loss[i] = sum_j K[i,j] * n[i] * n[j]
        loss_i = torch.sum(collision_rate, dim=2)  # Somme sur j: (ncells, nclasses)
        loss_j = torch.sum(collision_rate, dim=1)  # Somme sur i: (ncells, nclasses)

        Scoa = -loss_i - loss_j  # (ncells, nclasses)

        # ========== GAIN (scatter_add) ==========
        # Chaque collision (i,j) produit une particule qui est splittée entre k_low et k_high
        collision_flat = collision_rate.flatten()  # (ncells * nclasses * nclasses,)

        # Valeurs à ajouter dans k_low et k_high
        values_low = collision_flat * self._frac_low_flat
        values_high = collision_flat * (1.0 - self._frac_low_flat)

        # scatter_add: accumulation dans Scoa
        Scoa_flat = Scoa.reshape(-1)  # (ncells * nclasses,)
        Scoa_flat.scatter_add_(0, self._flat_idx_low, values_low)
        Scoa_flat.scatter_add_(0, self._flat_idx_high, values_high)

        return Scoa.reshape(ns.shape)  # (ncells, nclasses)
