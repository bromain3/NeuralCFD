from NeuralCFD.soot_chemistry.coagulation.DummyCoagulation import DummyCoagulation
import torch

class Fuchs(DummyCoagulation):

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

        # Correction de Cunningham
        kn = self.lambda_g / self.radii
        self.cc = 1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn))  # (nclasses,)

        ri = self.radii[None, :, None]  # (1, nclasses, 1)
        rj = self.radii[None, None, :]  # (1, 1, nclasses)
        cc_i = self.cc[None, :, None]
        cc_j = self.cc[None, None, :]

        self.geometricPart = 2 / 3 * self.kB * (ri + rj) * (cc_i / ri + cc_j / rj)

    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        Calcul du noyau de coagulation brownien (GPU)

        Formule de Fuchs (transition free molecular / continuum)
        """

        # Noyau de coagulation (approximation continuum simplifiée)
        # K = (2kBT/3μ) × (ri + rj) × (Cc_i/ri + Cc_j/rj)
        prefactor = (Ts / self.mu)[:, None, None]  # (ncells, 1, 1)

        K = prefactor * self.geometricPart  # (ncells, nclasses, nclasses)

        return K