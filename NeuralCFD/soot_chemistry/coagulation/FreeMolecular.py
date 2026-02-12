from NeuralCFD.soot_chemistry.coagulation.DummyCoagulation import DummyCoagulation
import torch


class FreeMolecular(DummyCoagulation):
    """
    Modèle de coagulation en régime moléculaire libre.
    Implémente exactement la formule de sootlib (coagulationModel_FM).

    Formule de Seinfeld and Pandis (2016), p.548:
    K_ij = K_fm * sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²

    où K_fm = eps_c * sqrt(π/2 * kB * T) * (6/(π*rho_soot))^(2/3)
    """

    def __init__(self, grid, RHO_SOOT, eps_c=2.2, FM_multiplier=9.0/2.0/2.2):
        """
        Args:
            grid: objet contenant les informations sur le maillage de tailles
            RHO_SOOT: densité de la suie (kg/m³)
            binGrowthFactor: facteur de croissance géométrique F entre bins
            eps_c: facteur de correction (défaut=2.2) coagulation constant/van der Waals enhancement factor: Harris and Kennedy CST 59:443-454 (1988) https://doi.org/10.1080/00102208808947110
            FM_multiplier: multiplicateur du taux FM (défaut=9 / 2 / 2.2)
                          pour modèle LL: FM_multiplier = 9/(2*eps_c)
        """
        super().__init__(grid, RHO_SOOT)

        self.eps_c = eps_c
        self.FM_multiplier = FM_multiplier

        # Précalcul des termes géométriques pour chaque bin
        self._precompute_geometry()

    def _precompute_geometry(self):
        """
        Précalcule les propriétés géométriques indépendantes de T.
        Correspond aux calculs dans getCoagulationSootRate de sootlib.
        """
        # Masses des particules (déjà calculées dans __init__)
        # self.mBins est en kg

        # mi^(1/3) pour chaque bin
        self.m_onethird = self.mBins ** (1.0 / 3.0)  # (nclasses,)

        # Précalcul des matrices pour tous les couples (i,j)
        # Forme: (1, nclasses, nclasses) pour broadcasting avec (ncells, 1, 1)

        mi = self.mBins[None, :, None]  # (1, nclasses, 1)
        mj = self.mBins[None, None, :]  # (1, 1, nclasses)

        mi_13 = self.m_onethird[None, :, None]
        mj_13 = self.m_onethird[None, None, :]

        # sqrt(1/mi + 1/mj)
        self.sqrt_inv_m_sum = torch.sqrt(1.0 / mi + 1.0 / mj)  # (1, nclasses, nclasses)

        # (mi^(1/3) + mj^(1/3))²
        m13_sum = mi_13 + mj_13
        self.m13_sum_squared = m13_sum * m13_sum  # (1, nclasses, nclasses)

    def getKfm(self, Ts):
        """
        Calcule le préfacteur K_fm dépendant de la température.

        Formule de sootlib:
        K_fm = FM_multiplier * eps_c * sqrt(π/2 * kB * T) * (6/(π*rho_soot))^(2/3)

        Args:
            Ts: températures (ncells,) en K

        Returns:
            K_fm: préfacteur (ncells,) en m^(5/3)/s
        """
        # Terme en température: sqrt(π/2 * kB * T)
        temp_term = torch.sqrt(0.5 * torch.pi * self.kB * Ts)  # (ncells,)

        # Terme géométrique: (6/(π*rho_soot))^(2/3)
        geom_term = (6.0 / (torch.pi * self.RHO_SOOT)) ** (2.0 / 3.0)  # scalaire

        # K_fm complet
        K_fm = self.FM_multiplier * self.eps_c * temp_term * geom_term  # (ncells,)

        return K_fm

    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcule la matrice du noyau de coagulation en régime moléculaire libre.

        Implémente exactement:
        K[i,j] = getKfm(T) * sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²

        Args:
            ns: concentrations en nombre (ncells, nclasses) [non utilisé pour FM]
            ys: fractions massiques (ncells, nspecies) [non utilisé pour FM]
            rhos: densités (ncells,) [non utilisé pour FM]
            Ts: températures (ncells,) en K

        Returns:
            K: matrice de noyau (ncells, nclasses, nclasses) en m³/s
        """
        # Calcul du préfacteur K_fm(T)
        K_fm = self.getKfm(Ts)  # (ncells,)

        # Reshape pour broadcasting: (ncells, 1, 1)
        K_fm = K_fm[:, None, None]

        # Noyau de coagulation:
        # K[icell, i, j] = K_fm[icell] * sqrt(1/mi + 1/mj) * (mi^(1/3) + mj^(1/3))²
        K = K_fm * self.sqrt_inv_m_sum * self.m13_sum_squared  # (ncells, nclasses, nclasses)

        return K

    def getCoagulationSootRate(self, state_T, m1, m2):
        """
        Version scalaire pour un seul couple de particules.
        Réplique exactement coagulationModel_FM::getCoagulationSootRate de sootlib.

        Args:
            state_T: température (scalaire) en K
            m1: masse de la particule 1 (scalaire) en kg
            m2: masse de la particule 2 (scalaire) en kg

        Returns:
            K12: taux de coagulation (scalaire) en m³/s
        """
        mmin = 2e-26  # masse minimale (comme dans sootlib) (mass of a carbon atom (kg))

        if m1 <= mmin or m2 <= mmin:
            return 0.0

        # Calcul de K_fm
        K_fm_scalar = self.getKfm(state_T)

        # (m1^(1/3) + m2^(1/3))²
        m13_m13 = m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)
        m13_m13_squared = m13_m13 * m13_m13

        # K = K_fm * sqrt(1/m1 + 1/m2) * (m1^(1/3) + m2^(1/3))²
        K12 = K_fm_scalar * torch.sqrt(1.0 / m1 + 1.0 / m2) * m13_m13_squared

        return K12