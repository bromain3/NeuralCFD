from NeuralCFD.soot_chemistry.nucleation.DummyNucleation import DummyNucleation
import torch

class LeungLindset(DummyNucleation):
    def __init__(self, grid, thermo, RHO_SOOT, C2H2Name='C2H2', H2Name='H2'):
        DummyNucleation.__init__(self, grid, RHO_SOOT)
        self.coefsSto = torch.zeros(grid.nspecies, device=grid.device, dtype=grid.dtype)
        self.coefsSto[thermo.thermoProp.species.index(C2H2Name)] = -1.0
        self.coefsSto[thermo.thermoProp.species.index(H2Name)] = 1.0
        self.C2H2idx = thermo.thermoProp.species.index(C2H2Name)
        self.MW_C2H2 = torch.tensor(26e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)


        # ============== PARAMÈTRES DE NUCLÉATION ==============
        print(f"incipient soot particule contains {self.C_min} carbon atoms. This model is designe to work best with incipient soot particles containing 100 atoms")



    def nucleationRate(self, ns, ys, rhos, Ts):
        """
        Nucléation de Leung & Lindstedt (1991)

        Équation SootLib:
        Rnuc = 0.1×10^5 × exp(-21100/T) × [C2H2] × 2Na/Cmin

        Unités:
        - [C2H2] en mol/m³ (concentration molaire)
        - Rnuc en #/(m³·s)

        Retourne:
            dndt: (ncells, nclasses) - source de nucléation en #/(m³·s)
        """
        # Conversion Y → concentration molaire (kmol/m³)
        # C = ρ × Y / MW / 1000
        C_C2H2 = rhos * ys[:, self.C2H2idx] / self.MW_C2H2 / 1000  # (ncells,)

        # Constante de vitesse (d'après code SootLib)
        k_nuc = 0.1e5 * torch.exp(-21_100.0 / Ts)  # m³/(kmol·s)  # constante originelle de Leunget Lindset
        # k_nuc = 0.1e5 * torch.exp(-41_100.0 / grid.cells.T)  # m³/(kmol·s)  # constante plus récente

        # Taux de nucléation (kmol/m³/s)
        R_nuc_molar = k_nuc * C_C2H2  # kmol/(m³·s)

        # Convertir en #/m³/s avec facteur 2Na/Cmin
        # D'après SootLib: return Rnuc * 2 * Na / state.cMin car Na est en #/kmol
        R_nuc = R_nuc_molar * 2.0 * (self.Na * 1000) / self.C_min  # #/(m³·s)

        return(R_nuc)

    def getStoechiometricCoefs(self):
        return(self.coefsSto)

