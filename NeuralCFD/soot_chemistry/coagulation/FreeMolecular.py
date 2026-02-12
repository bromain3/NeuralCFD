from NeuralCFD.soot_chemistry.coagulation.DummyCoagulation import DummyCoagulation
import torch


class FreeMolecular(DummyCoagulation):

    def __init__(self, grid, RHO_SOOT):
        DummyCoagulation.__init__(self, grid, RHO_SOOT)
        self.__precompute_geometry(grid)

    def __precompute_geometry(self, grid):
        """Précalcul des propriétés géométriques sur GPU."""
        v = grid.meanParticuleVolume  # (nclasses,) volumes en m³

        # Rayons des particules: r = (3V/4π)^(1/3)
        self.radii = (3 * v / (4 * torch.pi)) ** (1 / 3)  # (nclasses,)

        # Surface des particules: S = 4πr²
        self.surfaces = 4 * torch.pi * self.radii ** 2  # (nclasses,)

        # Masses des particules: m = ρ_soot × V
        self.masses = self.RHO_SOOT * v  # (nclasses,)

        # Précalcul des termes géométriques
        ri = self.radii[None, :, None]  # (1, nclasses, 1)
        rj = self.radii[None, None, :]  # (1, 1, nclasses)
        mi = self.masses[None, :, None]
        mj = self.masses[None, None, :]

        # (ri + rj)²
        self.r_sum_squared = (ri + rj) ** 2  # (1, nclasses, nclasses)

        # Masse réduite: 1/μ_ij = 1/mi + 1/mj
        self.inv_reduced_mass = 1 / mi + 1 / mj  # (1, nclasses, nclasses)

    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcul du noyau de coagulation en régime moléculaire libre (GPU)

        Formule de Fuchs pour free molecular:
        K_ij = sqrt(3kB/ρ_soot) × (ri + rj)² × sqrt(T/Vij)

        où Vij est le volume moyen = (Vi^(1/3) + Vj^(1/3))³

        Formule alternative plus courante:
        K_ij = sqrt(6πkB*T) × (ri + rj)² × sqrt(1/mi + 1/mj)
        """

        # Préfacteur: sqrt(6πkB*T)
        prefactor = torch.sqrt(6 * torch.pi * self.kB * Ts[:, None, None])  # (ncells, 1, 1)

        # Noyau de coagulation free molecular
        K = prefactor * self.r_sum_squared * torch.sqrt(self.inv_reduced_mass)  # (ncells, nclasses, nclasses)

        return K