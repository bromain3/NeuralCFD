from NeuralCFD.soot_chemistry.growth.DummyGrowth import DummyGrowth
import torch

class Lindset(DummyGrowth):
    def __init__(self, grid, thermo, RHO_SOOT, C2H2Name='C2H2', H2Name='H2'):
        DummyGrowth.__init__(self, grid, RHO_SOOT)
        self.coefsSto = torch.zeros(grid.nspecies, device=grid.device, dtype=grid.dtype)
        self.coefsSto[thermo.thermoProp.species.index(C2H2Name)] = -1.0
        self.coefsSto[thermo.thermoProp.species.index(H2Name)] = 1.0


        self.C2H2idx = thermo.thermoProp.species.index(C2H2Name)
        self.MW_C2H2 = torch.tensor(26e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)
        self.MW_C = torch.tensor(12e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)
        self.Na = torch.tensor(6.022e23, device=grid.device, dtype=grid.dtype)  # #/mol

        # Masse d'une particule dans une section donnée (kg)
        self.m_class = grid.meanParticuleVolume * self.RHO_SOOT

    def growthRate(self, ns, ys, rhos, Ts):
        """
        Croissance de Lindstedt (1994)

        Référence: Bockhorn (1994) pg. 417, Equation (27.36)
        "Simplified Soot Nucleation and Surface Growth Steps..."

        Équation SootLib:
        rGrowth = 750 × exp(-12100/T) × [C2H2] × 2 × MW_C

        C2H2 → 2C(soot) + H2

        Caractéristique clé de Lindstedt:
        - Le taux est INDÉPENDANT de la surface disponible
        - Proportionnel au nombre de particules M0 (implicite via ns)
        - Pas de dépendance en √S comme Leung & Lindstedt

        Unités:
        - [C2H2] en mol/m³
        - rGrowth en kg/(m²·s)
        - Retour: taux volumique (m³/s)

        Returns:
            growth_rate: (ncells, nclasses) - taux positif en m³/s
        """

        # Concentration molaire C2H2 (mol/m³)
        C_C2H2 = rhos * ys[:, self.C2H2idx] / self.MW_C2H2  # (ncells,)

        # Taux de croissance surfacique (kg/m²/s)
        # Formule exacte de SootLib - INDÉPENDANT de la surface
        rGrowth = (750.0 * torch.exp(-12_100.0 / Ts) *
                   C_C2H2 * 2.0 * self.MW_C)  # (ncells,)


        return rGrowth

    def getStoechiometricCoefs(self):
        return(self.coefsSto)

