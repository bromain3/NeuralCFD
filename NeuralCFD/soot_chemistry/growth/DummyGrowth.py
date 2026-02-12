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

        # Taille des sections (différence de masse entre bins consécutifs)
        # sectionSize[k] = mBins[k+1] - mBins[k]
        self.sectionSize = self.mBins[1:] - self.mBins[:-1]  # kg

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
        Calcul du terme source de croissance Sgrw équivalent au code C

        :param ns: [batch, nsoot] - concentrations en nombre de particules (#/m³)
        :return: dndt [batch, nsoot] - terme source (#/m³/s)
        """
        # Initialisation
        dndt = torch.zeros_like(ns)  # [batch, nsoot]

        # Surface totale par unité de volume pour chaque classe
        # Am2m3[k] = meanParticuleSurface[k] * ns[k]  (m²/m³)
        Am2m3 = self.meanParticuleSurface[None, :] * ns  # [batch, nsoot]

        # Taux de croissance volumétrique (m³/s)
        kGrw = self.growthRate(ns, ys, rhos, Ts)  # [batch, 1] ou scalaire

        # Calcul vectorisé des termes de flux
        # term[k] = kGrw_mass * Am2m3[k] / sectionSize[k] pour k=0..nsoot-2
        term = kGrw[:, None] * Am2m3[:, :-1] / self.sectionSize[None, :]  # [batch, nsoot-1]

        # Première classe (k=0) : sortie uniquement
        dndt[:, 0] = -term[:, 0]

        # Classes intermédiaires (k=1..nsoot-2) : entrée depuis k-1, sortie vers k+1
        if ns.shape[1] > 2:  # s'il y a des bins intermédiaires
            dndt[:, 1:-1] = term[:, :-1] - term[:, 1:]

        # Dernière classe (k=nsoot-1) :
        # - entrée depuis k-1 : term[nsoot-2]
        # - croissance interne : kGrw_mass * Am2m3[nsoot-1] / mBins[nsoot-1]
        dndt[:, -1] = term[:, -1] + kGrw * Am2m3[:, -1] / self.mBins[-1]  # le fait que les particules qui croissent dans la dernière classe augmente le nombre de particules modélise implicitement une fragmentation des grosses particules

        # ajouter une rétroaction sur tous les autres paramettres pour avoir un modèle complet
        pass

        return dndt