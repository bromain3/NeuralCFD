from abc import ABC, abstractmethod
import torch
import math


class DummyCoagulation(ABC):
    """
    Classe de base pour les modèles de coagulation, implémentant la méthode SECT
    (Sectional) de sootlib avec splitting pour conservation de la masse et du nombre.
    """

    def __init__(self, grid, RHO_SOOT):
        """
        Args:
            grid: objet contenant les informations sur le maillage de tailles
            RHO_SOOT: densité de la suie (kg/m³)
            binGrowthFactor: facteur de croissance géométrique F entre bins
        """
        self.kB = torch.tensor(1.380649e-23, device=grid.device, dtype=grid.dtype)  # J/K
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)
        self.binGrowthFactor = grid.binGrowthFactor

        # Précalcul de 1/ln(F) pour localiser les bins cibles
        self.ilnF = 1.0 / math.log(self.binGrowthFactor)

        # Masse d'une particule dans chaque section (kg)
        # Correspond à mBins dans sootlib
        self.mBins = grid.meanParticuleVolume * self.RHO_SOOT  # (nclasses,)

        self.nclasses = len(self.mBins)

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
        Calcule le terme source de coagulation pour toutes les cellules.
        Implémente exactement l'algorithme SECT de sootlib avec splitting.

        Args:
            ns: concentrations en nombre (ncells, nclasses) en #/m³
            K: matrice de noyau (ncells, nclasses, nclasses) en m³/s

        Returns:
            Scoa: termes source de coagulation (ncells, nclasses) en #/m³/s
        """
        ncells, nclasses = ns.shape
        Scoa = torch.zeros_like(ns)

        # Boucle sur toutes les cellules (vectorisé)
        # Boucles sur les classes (explicites comme dans sootlib)

        for i in range(nclasses):
            for j in range(i, nclasses):  # Triangle supérieur + diagonale

                # ========== PERTE: collision i+j retire des bins i et j ==========
                # K12 * n[i] * n[j]
                K12 = self.getCoagulationSootRate(Ts, self.mBins[i], self.mBins[j])  # (ncells,)
                term = K12 * ns[:, i] * ns[:, j]  # (ncells,)

                Scoa[:, i] -= term
                if j > i:  # Symétrie: termes sous la diagonale
                    Scoa[:, j] -= term

                # ========== GAIN: mi + mj -> splitting entre floor(k) et ceil(k) ==========
                # Facteur 0.5 pour éviter double comptage sur la diagonale
                if i == j:
                    term = term * 0.5

                # Masse combinée
                m_combined = self.mBins[i] + self.mBins[j]  # scalaire

                # Trouver l'indice k tel que mBins[0] * F^k ≈ m_combined
                # k = log((mi + mj) / m0) / log(F)
                k_float = self.ilnF * torch.log(m_combined / self.mBins[0])
                k = int(k_float.item()) if isinstance(k_float, torch.Tensor) else int(k_float)

                # Clamp k dans [0, nclasses-1]
                if k >= nclasses:
                    k = nclasses - 1

                if k == nclasses - 1:
                    # Dernière bin: pas de splitting, conservation de la masse
                    # Gain = term * (mi + mj) / mk
                    Scoa[:, k] += term * (m_combined / self.mBins[k])
                else:
                    # Splitting entre bins k et k+1 pour conserver masse ET nombre
                    # Fraction dans bin k:
                    # Xk = (m[k+1] - m_combined) / (m[k+1] - m[k])
                    m_k = self.mBins[k]
                    m_k1 = self.mBins[k + 1]
                    Xk = (m_k1 - m_combined) / (m_k1 - m_k)

                    Scoa[:, k] += Xk * term
                    Scoa[:, k + 1] += (1.0 - Xk) * term

        return Scoa
