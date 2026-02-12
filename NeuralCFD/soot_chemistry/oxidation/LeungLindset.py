from NeuralCFD.soot_chemistry.oxidation.DummyOxidation import DummyOxidation
import torch

class LeungLindset(DummyOxidation):
    def __init__(self, grid, thermo, RHO_SOOT, O2Name='O2', COName='CO'):
        DummyOxidation.__init__(self, grid, RHO_SOOT)
        self.coefsSto = torch.zeros(grid.nspecies, device=grid.device, dtype=grid.dtype)
        self.coefsSto[thermo.thermoProp.species.index(O2Name)] = -0.5
        self.coefsSto[thermo.thermoProp.species.index(COName)] = 1.0
        self.O2idx = thermo.thermoProp.species.index(O2Name)
        self.MW_O2 = torch.tensor(32e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)
        self.MW_C = torch.tensor(12e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)



    def oxidationRate(self, ns, ys, rhos, Ts):
        """
        Oxydation de Leung & Lindstedt (1991)

        Équation SootLib:
        rOxi = 0.1×10^5 × √T × exp(-19680/T) × [O2] × MW_C

        Unités:
        - [O2] en mol/m³
        - rOxi en kg/(m²·s)
        - L = rOxi × S / ρ_s en m³/s

        Retourne:
            L: (ncells, nclasses) - taux d'oxydation volumique en m³/s
        """
        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_O2 = rhos * ys[:, self.O2idx] / self.MW_O2 / 1000 # (ncells,)

        # oxydation rate (kg/m2/s)
        rSoot = 0.1e5 * torch.sqrt(Ts) * torch.exp(-19_680.0 / Ts) * C_O2 * (self.MW_C * 1000)


        return rSoot

    def getStoechiometricCoefs(self):
        return(self.coefsSto)

