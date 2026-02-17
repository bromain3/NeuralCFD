from NeuralCFD.soot_chemistry.oxidation.DummyOxidation import DummyOxidation
import torch


class HACA(DummyOxidation):
    """
    HACA oxidation model.

    Based on:
    - Appel, Bockhorn, Frenklach, Combustion and Flame 121:122-136 (2000)
    - Frenklach and Wang (1990), 23rd Symposium, pp. 1559-1566

    Reactions:
    5. Cs*  + O2   ==> 2CO + (products)
    6. Cs-H + OH   ==> CO + (products)

    Balance for reaction 6: C + OH ==> CO + H
    Overall balance: 3C + O2 + OH ==> 3CO + H
    """

    def __init__(self, grid, thermo, RHO_SOOT,
                 O2Name='O2', OHName='OH', HName='H',
                 H2Name='H2', H2OName='H2O', C2H2Name='C2H2',
                 COName='CO'):
        DummyOxidation.__init__(self, grid, RHO_SOOT)

        # Stoichiometric coefficients for species balance
        self.coefsSto = torch.zeros(grid.nspecies, device=grid.device, dtype=grid.dtype)
        self.coefsSto[thermo.thermoProp.species.index(O2Name)] = -1
        self.coefsSto[thermo.thermoProp.species.index(OHName)] = -1
        self.coefsSto[thermo.thermoProp.species.index(HName)] = 1
        self.coefsSto[thermo.thermoProp.species.index(COName)] = 3
        # Carbon balance: -3 (handled separately in oxidation rate)

        # Species indices
        self.O2idx = thermo.thermoProp.species.index(O2Name)
        self.OHidx = thermo.thermoProp.species.index(OHName)
        self.Hidx = thermo.thermoProp.species.index(HName)
        self.H2idx = thermo.thermoProp.species.index(H2Name)
        self.H2Oidx = thermo.thermoProp.species.index(H2OName)
        self.C2H2idx = thermo.thermoProp.species.index(C2H2Name)

        # Molecular weights (kg/mol)
        self.MW_O2 = torch.tensor(32e-3, device=grid.device, dtype=grid.dtype)
        self.MW_OH = torch.tensor(17e-3, device=grid.device, dtype=grid.dtype)
        self.MW_H = torch.tensor(1e-3, device=grid.device, dtype=grid.dtype)
        self.MW_H2 = torch.tensor(2e-3, device=grid.device, dtype=grid.dtype)
        self.MW_H2O = torch.tensor(18e-3, device=grid.device, dtype=grid.dtype)
        self.MW_C2H2 = torch.tensor(26e-3, device=grid.device, dtype=grid.dtype)
        self.MW_C = torch.tensor(12e-3, device=grid.device, dtype=grid.dtype)

        # Constants
        self.Na = torch.tensor(6.022e23, device=grid.device, dtype=grid.dtype)  # #/mol
        self.R = torch.tensor(1.9872036e-3, device=grid.device, dtype=grid.dtype)  # kcal/(mol·K)
        self.Rg = torch.tensor(8.314, device=grid.device, dtype=grid.dtype)  # J/(mol·K)
        self.chi_CH = torch.tensor(2.3e15, device=grid.device, dtype=grid.dtype)  # C-H sites/cm²
        self.pi = torch.tensor(3.141592653589793, device=grid.device, dtype=grid.dtype)

        # Masse d'une particule dans une section donnée (kg)
        self.m_class = grid.meanParticuleVolume * self.RHO_SOOT

    def oxidationRate(self, ns, ys, rhos, Ts):
        """
        HACA oxidation rate calculation.

        Combines two oxidation pathways:
        - RO2: Oxidation via O2 (reaction 5)
        - ROH: Oxidation via OH (reaction 6)

        Args:
            ns: (ncells, nclasses) - number density (#/m³)
            ys: (ncells, nspecies) - mass fractions
            rhos: (ncells,) - gas density (kg/m³)
            Ts: (ncells,) - temperature (K)

        Returns:
            rSoot: (ncells,) - soot oxidation rate (kg/m²·s)
        """
        # Calculate moments M0 (#/m³) and M1 (kg/m³)
        M0 = torch.sum(ns, dim=1)  # (ncells,)
        M1 = torch.sum(ns * self.m_class, dim=1)  # (ncells,)

        # Convert mass fractions to molar concentrations (kmol/m³)
        # C = ρ × Y / MW / 1000 (kmol/m³)
        C_O2 = rhos * ys[:, self.O2idx] / self.MW_O2 / 1000  # kmol/m³
        C_OH = rhos * ys[:, self.OHidx] / self.MW_OH / 1000
        C_H = rhos * ys[:, self.Hidx] / self.MW_H / 1000
        C_H2 = rhos * ys[:, self.H2idx] / self.MW_H2 / 1000
        C_H2O = rhos * ys[:, self.H2Oidx] / self.MW_H2O / 1000
        C_C2H2 = rhos * ys[:, self.C2H2idx] / self.MW_C2H2 / 1000

        # Calculate RT (kcal/mol)
        RT = self.R * Ts  # (ncells,)

        # Calculate a_param and b_param for alpha
        a_param = 33.167 - 0.0154 * Ts  # (ncells,)
        b_param = -2.5786 + 0.00112 * Ts  # (ncells,)

        # Calculate raw HACA reaction rates (1/s) - same as in growth
        fR1 = 4.2e13 * torch.exp(-13.0 / RT) * C_H / 1000  # (ncells,)
        rR1 = 3.9e12 * torch.exp(-11.0 / RT) * C_H2 / 1000
        fR2 = 1.0e10 * torch.pow(Ts, 0.734) * torch.exp(-1.43 / RT) * C_OH / 1000
        rR2 = 3.68e8 * torch.pow(Ts, 1.139) * torch.exp(-17.1 / RT) * C_H2O / 1000
        fR3 = 2.0e13 * C_H / 1000
        fR4 = 8.00e7 * torch.pow(Ts, 1.56) * torch.exp(-3.8 / RT) * C_C2H2 / 1000
        fR5 = 2.2e12 * torch.exp(-7.5 / RT) * C_O2 / 1000

        # Steady state calculation of chi_S (soot radical sites/cm²)
        denom = rR1 + rR2 + fR3 + fR4 + fR5
        chi_S = torch.zeros_like(Ts)
        mask_denom = denom > 0
        chi_S[mask_denom] = 2 * self.chi_CH * (fR1[mask_denom] + fR2[mask_denom]) / denom[mask_denom]

        # Calculate alpha (steric factor)
        alpha = torch.ones_like(Ts)
        mask_M0 = M0 > 0
        if mask_M0.any():
            log_ratio = torch.log10(M1[mask_M0] / M0[mask_M0])
            alpha_calc = torch.tanh(a_param[mask_M0] / log_ratio + b_param[mask_M0])
            # If alpha < 0, set to 1.0
            alpha[mask_M0] = torch.where(alpha_calc < 0,
                                         torch.ones_like(alpha_calc),
                                         alpha_calc)

        # ===== Oxidation via O2 (reaction 5) =====
        # Calculate reaction rate at sites (reactions/m²/s)
        rxns_s = fR5 * alpha * chi_S * 1e4  # reactions/m²/s
        # Convert to kg/m²/s (each site loses 2 carbons)
        RO2 = 2.0 * rxns_s * (self.MW_C) / self.Na  # kg/m²/s

        # ===== Oxidation via OH (reaction 6) =====
        # ROH = 0.13 * [OH] * sqrt(Rg*T/(2*pi*MW_OH)) * MW_C
        # [OH] in mol/cm³, need to convert to mol/m³ for consistency

        # Molecular flux term: sqrt(Rg*T/(2*pi*MW_OH))
        # MW_OH in kg/mol, Rg in J/(mol·K)
        flux_term = torch.sqrt(self.Rg * Ts / (2.0 * self.pi * self.MW_OH))  # m/s

        # ROH in kg/m²/s
        ROH = 0.13 * C_OH * flux_term * (self.MW_C * 1000)  # kg/m²/s

        # Total oxidation rate
        rSoot = RO2 + ROH  # kg/m²/s

        return rSoot

    def getStoechiometricCoefs(self):
        return self.coefsSto