import torch
from abc import ABC, abstractmethod

#TODO: take into account the section/dx variation. (if we have a mooving mesh, it will appear as a separate strang splitting operator or in the transport methodes, but no change to the chemistry solver)
class DummyGrowth(ABC):
    def __init__(self, grid, RHO_SOOT):
        self.meanParticuleVolume = grid.meanParticuleVolume
        self.meanParticuleSurface = grid.meanParticuleSurface

        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)  # kg/m³ (densité suie)
        # Conversion volume → masse pour les bins
        # mBins[k] = volume[k] * RHO_SOOT
        self.mBins = self.meanParticuleVolume * self.RHO_SOOT  # kg

    @abstractmethod
    def growthRate(self, ns, ys, rhos, Ts):
        """
        :param grid: the computation grid as defined in NeuralCFD.grid.grid
        :return: The growth rate in m3/s for all the classes
        """
        raise NotImplementedError('Any soot growth class inheriting from DummyGrowth must implement a growthRate method')

    @abstractmethod
    def getStoechiometricCoefs(self):
        """
        :return: the stoechiometric coefficient of the nucleation reaction (for consuming gaseous reactants)
        """
        raise NotImplementedError('Any soot growth class inheriting from DummyGrowth must implement a getStoechiometricCoefs method.')

    def dndt(self, ns, ys, rhos, Ts):
        """
        Calcul du terme source de croissance Sgrw équivalent au code C++ SECT

        :param ns: [batch, nsoot] - concentrations en nombre de particules (#/m³)
        :param ys: [batch, n_species] - fractions massiques des espèces gazeuses
        :param rhos: [batch] - densité du mélange gazeux (kg/m³)
        :param Ts: [batch] - température (K)
        :return: dndt [batch, nsoot] - terme source (#/m³/s)
        """
        nsoot = ns.shape[1]

        # Initialisation
        dndt = torch.zeros_like(ns)  # [batch, nsoot]

        # Taux de croissance massique (kg/m²/s)
        kGrw = self.growthRate(ns, ys, rhos, Ts)  # [batch]

        # Surface totale par unité de volume pour chaque classe
        # Am2m3[k] = meanParticuleSurface[k] * ns[k]  (m²/m³)
        Am2m3 = self.meanParticuleSurface[None, :] * ns  # [batch, nsoot]

        # k=0: première classe (sortie uniquement)
        k = 0
        term = kGrw * Am2m3[:, k] / (self.mBins[k + 1] - self.mBins[k])  # [batch]
        dndt[:, k] = -term

        # k=1 à nsoot-2: classes intermédiaires (entrée depuis k-1, sortie vers k+1)
        for k in range(1, nsoot - 1):
            dndt[:, k] = term  # entrée depuis k-1
            term = kGrw * Am2m3[:, k] / (self.mBins[k + 1] - self.mBins[k])  # [batch]
            dndt[:, k] -= term  # sortie vers k+1

        # k=nsoot-1: dernière classe (entrée depuis k-1 + croissance interne)
        k = nsoot - 1
        dndt[:, k] = term + kGrw * Am2m3[:, k] / self.mBins[k]

        # ajouter une rétroaction sur tous les autres paramettres pour avoir un modèle complet
        pass

        return dndt