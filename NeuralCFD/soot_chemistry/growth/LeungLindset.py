from NeuralCFD.soot_chemistry.growth.DummyGrowth import DummyGrowth
import torch

class LeungLindset(DummyGrowth):
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
        Croissance de Leung & Lindstedt (1991) - VERSION MOMENTS

        Équation SootLib (pour modèle à moments):
        rSoot = 0.6×10^4 × exp(-12100/T) × [C2H2] / √(A_m2m3) × 2 × MW_C

        Unités:
        - [C2H2] en mol/m³
        - rSoot en kg/(m²·s)
        - G = rSoot × S / ρ_s en m³/s

        Retourne:
            G: (ncells, nclasses) - taux de croissance volumique en kg/m2/s
        """
        #TODO: adapter pour un modèle sectionnel en faisant un calcul par classe pour être plus précis

        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_C2H2 = rhos * ys[:, self.C2H2idx] / self.MW_C2H2 / 1000 # (ncells,)

        # Total number of particles per unit volume #/m3
        M0 = torch.sum(ns, dim=1)
        M1 = torch.sum(ns * self.m_class, dim=1)

        # soot surface available for reaction (m2_soot/m3_total)
        Am2m3 = torch.zeros_like(Ts)
        mask = M0 > 0.0
        Am2m3[mask] = torch.pi * torch.pow(6.0 / (torch.pi * self.RHO_SOOT) * M1[mask] / M0[mask], 2.0 / 3.0) * M0[mask]

        # soot growth rate (kg/m2/s)
        rSoot = torch.zeros_like(Ts)
        mask = Am2m3 > 0.0
        rSoot[mask] = 0.6e4 * torch.exp(-12_100.0 / Ts[mask]) * C_C2H2[mask] / torch.sqrt(Am2m3[mask]) * 2 * (self.MW_C * 1000)

        return rSoot

    def getStoechiometricCoefs(self):
        return(self.coefsSto)

