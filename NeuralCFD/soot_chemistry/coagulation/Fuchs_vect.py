from NeuralCFD.soot_chemistry.coagulation.DummyCoagulation_vect import DummyCoagulation_vect
import torch


class Fuchs_vect(DummyCoagulation_vect):
    """
    Modèle de coagulation FUCHS - VERSION VECTORISÉE
    Transition douce entre les régimes moléculaire libre (FM) et continuum (C).

    Formule de Seinfeld and Pandis (2016), p.548:
    K_ij = 2π(Di + Dj)(di + dj) / [
        (di + dj)/(di + dj + 2*sqrt(gi² + gj²)) +
        8/(eps_c*FM_multiplier) * (Di + Dj)/(sqrt(ci² + cj²)*(di + dj))
    ]

    Où:
    - di, dj: diamètres des particules
    - Di, Dj: coefficients de diffusion (avec correction de Cunningham)
    - ci, cj: vitesses thermiques moyennes
    - gi, gj: termes géométriques de transition
    - mfp: libre parcours moyen du gaz

    OPTIMISATIONS:
    - Précalcul des propriétés géométriques indépendantes de l'état du gaz
    - Vectorisation complète pour toutes les paires (i,j)
    - Pas de boucles Python explicites
    """

    def __init__(self, grid, RHO_SOOT, eps_c=2.2, FM_multiplier=9.0 / 2.0 / 2.2):
        """
        Args:
            grid: objet contenant les informations sur le maillage de tailles
            RHO_SOOT: densité de la suie (kg/m³)
            eps_c: facteur de correction (défaut=2.2)
                   van der Waals enhancement factor
            FM_multiplier: multiplicateur du taux FM (défaut=9 / 2 / 2.2)
        """
        super().__init__(grid, RHO_SOOT)

        self.eps_c = eps_c
        self.FM_multiplier = FM_multiplier

        # Constantes pour la correction de Cunningham (Seinfeld p.372 eq.9.34)
        # Valables pour l'air à 298 K, 1 atm
        self.Cc_A = 1.257
        self.Cc_B = 0.4
        self.Cc_C = 1.1

        # Précalcul des termes géométriques constants
        self._precompute_geometry()

    def _precompute_geometry(self):
        """
        Précalcule les propriétés géométriques constantes (indépendantes de T, P, mu).

        Précalculs:
        1. di = (6*mi / (π*ρ))^(1/3) pour chaque bin
        2. Matrices (di + dj) pour toutes les paires (i,j)
        """
        # Diamètre de chaque particule: di = (6*mi / (π*ρ))^(1/3)
        self.dBins = ((6.0 * self.mBins) / (torch.pi * self.RHO_SOOT)) ** (1.0 / 3.0)  # (nclasses,)

        # Matrices pour tous les couples (i,j)
        di = self.dBins[:, None]  # (nclasses, 1)
        dj = self.dBins[None, :]  # (1, nclasses)
        mi = self.mBins[:, None]  # (nclasses, 1)
        mj = self.mBins[None, :]  # (1, nclasses)

        # Sommes des diamètres: (di + dj)
        self.d_sum = di + dj  # (nclasses, nclasses)

        # Inverses des masses pour vitesse thermique: 1/mi, 1/mj
        self.inv_mi = 1.0 / mi  # (nclasses, 1)
        self.inv_mj = 1.0 / mj  # (1, nclasses)

        # Constantes numériques
        self.sqrt2 = torch.sqrt(torch.tensor(2.0))
        self.pi_times_8 = 8.0 * torch.pi

    def _compute_cunningham_correction(self, Kn):
        """
        Calcule le facteur de correction de Cunningham.
        Seinfeld p.372 eq.9.34: Cc = 1 + Kn * (1.257 + 0.4*exp(-1.1/Kn))

        Args:
            Kn: nombre de Knudsen (peut être un tenseur de n'importe quelle forme)

        Returns:
            Cc: facteur de correction (même shape que Kn)
        """
        return 1.0 + Kn * (self.Cc_A + self.Cc_B * torch.exp(-self.Cc_C / Kn))

    def _compute_diffusion_coefficients(self, Ts, muGas, mfp):
        """
        Calcule les coefficients de diffusion avec correction de Cunningham.

        D = kB*T/(3*π*mu) * Cc/d

        Args:
            Ts: températures (ncells,) en K
            muGas: viscosité dynamique du gaz (ncells,) en Pa.s
            mfp: libre parcours moyen du gaz (ncells,) en m

        Returns:
            D: coefficients de diffusion (ncells, nclasses) en m²/s
        """
        ncells = Ts.shape[0]

        # Nombre de Knudsen pour chaque cellule et chaque classe
        # Kn = 2*mfp/d
        Kn = 2.0 * mfp[:, None] / self.dBins[None, :]  # (ncells, nclasses)

        # Correction de Cunningham
        Cc = self._compute_cunningham_correction(Kn)  # (ncells, nclasses)

        # Coefficient de diffusion: D = kB*T/(3*π*mu) * Cc/d
        D = (self.kB * Ts[:, None]) / (3.0 * torch.pi * muGas[:, None]) * Cc / self.dBins[None, :]

        return D  # (ncells, nclasses)

    def _compute_thermal_velocities(self, Ts):
        """
        Calcule les vitesses thermiques moyennes.

        c = sqrt(8*kB*T / (π*m))

        Args:
            Ts: températures (ncells,) en K

        Returns:
            c: vitesses thermiques (ncells, nclasses) en m/s
        """
        # c = sqrt(8*kB*T / (π*m))
        c = torch.sqrt((8.0 * self.kB * Ts[:, None]) / (torch.pi * self.mBins[None, :]))

        return c  # (ncells, nclasses)

    def _compute_g_terms(self, D, c, d):
        """
        Calcule les termes géométriques g pour la transition.

        Formule de Fuchs:
        g = sqrt(2)/(3*d*l) * ((d+l)³ - (d² + l²)^(3/2)) - sqrt(2)*d

        où l = 8*D/(π*c)

        Args:
            D: coefficients de diffusion (ncells, nclasses)
            c: vitesses thermiques (ncells, nclasses)
            d: diamètres (nclasses,) - on utilise self.dBins

        Returns:
            g: termes g (ncells, nclasses)
        """
        # l = 8*D/(π*c)
        l = (8.0 * D) / (torch.pi * c)  # (ncells, nclasses)

        # d pour broadcasting
        d = self.dBins[None, :]  # (1, nclasses)

        # Termes pour la formule
        d_plus_l = d + l
        d_squared_plus_l_squared = d * d + l * l

        # g = sqrt(2)/(3*d*l) * ((d+l)³ - (d² + l²)^(3/2)) - sqrt(2)*d
        g = (self.sqrt2 / (3.0 * d * l)) * (
                d_plus_l ** 3.0 - d_squared_plus_l_squared ** 1.5
        ) - self.sqrt2 * d

        return g  # (ncells, nclasses)

    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcule la matrice du noyau de coagulation FUCHS - VERSION ENTIÈREMENT VECTORISÉE

        Implémente:
        K[icell, i, j] = 2π(Di + Dj)(di + dj) / [
            (di + dj)/(di + dj + 2*sqrt(gi² + gj²)) +
            8/(eps_c*FM_multiplier) * (Di + Dj)/(sqrt(ci² + cj²)*(di + dj))
        ]

        Args:
            ns: concentrations en nombre (ncells, nclasses) [non utilisé]
            ys: fractions massiques (ncells, nspecies)
            rhos: densités (ncells,) en kg/m³
            Ts: températures (ncells,) en K

        Returns:
            K: matrice de noyau (ncells, nclasses, nclasses) en m³/s
        """
        ncells = Ts.shape[0]

        # Calcul de la viscosité dynamique du gaz (dépend de T)
        # Loi de Sutherland ou approximation simple: mu ~ T^(1/2)
        # Ici on utilise une approximation simple (à adapter selon votre modèle)
        # mu_ref = 1.716e-5 Pa.s à T_ref = 273 K
        T_ref = 273.0
        mu_ref = 1.716e-5
        S = 110.4  # Constante de Sutherland pour l'air
        muGas = mu_ref * ((Ts / T_ref) ** 1.5) * ((T_ref + S) / (Ts + S))  # (ncells,)

        # Calcul du libre parcours moyen du gaz
        # mfp = mu * sqrt(π/(2*rho*R*T)) où R est la constante des gaz pour l'air
        # Approximation: mfp ~ mu / (rho * sqrt(kB*T/m_molecule))
        # Formule simplifiée: mfp ~ kB*T / (sqrt(2)*π*d_molecule²*P)
        # Ici on utilise une formule standard:
        # mfp = (2*mu) / (P * sqrt(8*M/(π*R*T)))
        # Ou plus simplement pour l'air: mfp ~ 6.8e-8 * T / P avec P en Pa

        # Pression (en supposant gaz parfait: P = rho * R_specific * T)
        # R_specific pour l'air ~ 287 J/(kg.K)
        R_specific = 287.0
        P = rhos * R_specific * Ts  # (ncells,) en Pa

        # Libre parcours moyen (formule approchée)
        mfp = (2.0 * muGas) / (P * torch.sqrt(8.0 * 0.029 / (torch.pi * 8.314 * Ts)))  # (ncells,)

        # 1. Coefficients de diffusion
        D = self._compute_diffusion_coefficients(Ts, muGas, mfp)  # (ncells, nclasses)

        # 2. Vitesses thermiques
        c = self._compute_thermal_velocities(Ts)  # (ncells, nclasses)

        # 3. Termes g
        g = self._compute_g_terms(D, c, self.dBins)  # (ncells, nclasses)

        # Maintenant, on construit les matrices pour toutes les paires (i,j)

        # Broadcasting: (ncells, nclasses, 1) et (ncells, 1, nclasses)
        Di = D[:, :, None]  # (ncells, nclasses, 1)
        Dj = D[:, None, :]  # (ncells, 1, nclasses)

        ci = c[:, :, None]  # (ncells, nclasses, 1)
        cj = c[:, None, :]  # (ncells, 1, nclasses)

        gi = g[:, :, None]  # (ncells, nclasses, 1)
        gj = g[:, None, :]  # (ncells, 1, nclasses)

        # Sommes et produits
        D_sum = Di + Dj  # (ncells, nclasses, nclasses)
        c_squared_sum = ci * ci + cj * cj  # (ncells, nclasses, nclasses)
        g_squared_sum = gi * gi + gj * gj  # (ncells, nclasses, nclasses)

        # d_sum est précalculé: (nclasses, nclasses)
        d_sum = self.d_sum[None, :, :]  # (1, nclasses, nclasses)

        # Numérateur: 2*π*(Di + Dj)*(di + dj)
        numerator = 2.0 * torch.pi * D_sum * d_sum  # (ncells, nclasses, nclasses)

        # Dénominateur: deux termes
        # Terme 1: (di + dj) / (di + dj + 2*sqrt(gi² + gj²))
        term1 = d_sum / (d_sum + 2.0 * torch.sqrt(g_squared_sum))  # (ncells, nclasses, nclasses)

        # Terme 2: 8/(eps_c*FM_multiplier) * (Di + Dj) / (sqrt(ci² + cj²) * (di + dj))
        term2 = (8.0 / (self.eps_c * self.FM_multiplier)) * D_sum / (
                torch.sqrt(c_squared_sum) * d_sum
        )  # (ncells, nclasses, nclasses)

        denominator = term1 + term2  # (ncells, nclasses, nclasses)

        # Noyau de coagulation
        K = numerator / denominator  # (ncells, nclasses, nclasses)

        return K