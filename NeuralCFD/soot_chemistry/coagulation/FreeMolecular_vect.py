from NeuralCFD.soot_chemistry.coagulation.DummyCoagulation_vect import DummyCoagulation_vect
import torch


class FreeMolecular_vect(DummyCoagulation_vect):
    """
    Modèle de coagulation en régime moléculaire libre - VERSION VECTORISÉE
    Implémente exactement la formule de sootlib (coagulationModel_FM).

    Formule de Seinfeld and Pandis (2016), p.548:
    K_ij = K_fm * sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²

    où K_fm = eps_c * sqrt(π/2 * kB * T) * (6/(π*rho_soot))^(2/3)

    OPTIMISATIONS:
    - Tous les termes géométriques indépendants de T sont précalculés dans __init__
    - compute_kernel_matrix() est entièrement vectorisé
    - Pas de boucles Python explicites
    """

    def __init__(self, grid, RHO_SOOT, eps_c=2.2, FM_multiplier=9.0 / 2.0 / 2.2):
        """
        Args:
            grid: objet contenant les informations sur le maillage de tailles
            RHO_SOOT: densité de la suie (kg/m³)
            eps_c: facteur de correction (défaut=2.2)
                   van der Waals enhancement factor: Harris and Kennedy CST 59:443-454 (1988)
            FM_multiplier: multiplicateur du taux FM (défaut=9 / 2 / 2.2)
                          pour modèle LL: FM_multiplier = 9/(2*eps_c)
        """
        super().__init__(grid, RHO_SOOT)

        self.eps_c = eps_c
        self.FM_multiplier = FM_multiplier

        # Précalcul des termes géométriques pour toutes les paires (i,j)
        self._precompute_geometry()

    def _precompute_geometry(self):
        """
        Précalcule les propriétés géométriques indépendantes de T.
        Tous les calculs sont vectorisés pour éviter les boucles.

        Précalculs:
        1. mi^(1/3) pour chaque bin
        2. sqrt(1/mi + 1/mj) pour toutes les paires (i,j)
        3. (mi^(1/3) + mj^(1/3))² pour toutes les paires (i,j)
        """
        # mi^(1/3) pour chaque bin
        self.m_onethird = self.mBins ** (1.0 / 3.0)  # (nclasses,)

        # Terme géométrique constant: (6/(π*rho_soot))^(2/3)
        self.geom_term = (6.0 / (torch.pi * self.RHO_SOOT)) ** (2.0 / 3.0)  # scalaire

        # Matrices pour tous les couples (i,j)
        # Shape: (nclasses, nclasses) pour broadcasting avec (ncells, 1, 1)

        mi = self.mBins[:, None]  # (nclasses, 1)
        mj = self.mBins[None, :]  # (1, nclasses)

        mi_13 = self.m_onethird[:, None]  # (nclasses, 1)
        mj_13 = self.m_onethird[None, :]  # (1, nclasses)

        # sqrt(1/mi + 1/mj)
        self.sqrt_inv_m_sum = torch.sqrt(1.0 / mi + 1.0 / mj)  # (nclasses, nclasses)

        # (mi^(1/3) + mj^(1/3))²
        m13_sum = mi_13 + mj_13
        self.m13_sum_squared = m13_sum * m13_sum  # (nclasses, nclasses)

        # Produit géométrique complet (indépendant de T):
        # geom[i,j] = sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²
        self.geom_matrix = self.sqrt_inv_m_sum * self.m13_sum_squared  # (nclasses, nclasses)

    def _compute_Kfm_prefactor(self, Ts):
        """
        Calcule le préfacteur K_fm dépendant de la température - VERSION VECTORISÉE

        Formule de sootlib:
        K_fm = FM_multiplier * eps_c * sqrt(π/2 * kB * T) * (6/(π*rho_soot))^(2/3)

        Args:
            Ts: températures (ncells,) en K

        Returns:
            K_fm: préfacteur (ncells,) en unités appropriées pour K
        """
        # Terme en température: sqrt(π/2 * kB * T)
        temp_term = torch.sqrt(0.5 * torch.pi * self.kB * Ts)  # (ncells,)

        # K_fm complet: FM_multiplier * eps_c * sqrt(π/2 * kB * T) * (6/(π*rho_soot))^(2/3)
        K_fm = self.FM_multiplier * self.eps_c * temp_term * self.geom_term  # (ncells,)

        return K_fm

    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcule la matrice du noyau de coagulation - VERSION ENTIÈREMENT VECTORISÉE

        Implémente exactement:
        K[icell, i, j] = K_fm(T[icell]) * sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²

        Tous les termes géométriques sont précalculés, seul K_fm(T) est calculé à chaque appel.

        Args:
            ns: concentrations en nombre (ncells, nclasses) [non utilisé pour FM]
            ys: fractions massiques (ncells, nspecies) [non utilisé pour FM]
            rhos: densités (ncells,) [non utilisé pour FM]
            Ts: températures (ncells,) en K

        Returns:
            K: matrice de noyau (ncells, nclasses, nclasses) en m³/s
        """
        # Calcul du préfacteur K_fm(T)
        K_fm = self._compute_Kfm_prefactor(Ts)  # (ncells,)

        # Reshape pour broadcasting: (ncells, 1, 1)
        K_fm = K_fm[:, None, None]

        # Noyau de coagulation:
        # K[icell, i, j] = K_fm[icell] * geom_matrix[i, j]
        # où geom_matrix = sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²
        K = K_fm * self.geom_matrix[None, :, :]  # (ncells, nclasses, nclasses)

        return K
