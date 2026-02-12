import torch
from abc import ABC, abstractmethod

#TODO: take into account the section/dx variation. (if we have a mooving mesh, it will appear as a separate strang splitting operator or in the transport methodes, but no change to the chemistry solver)

class DummyOxidation(ABC):
    def __init__(self, grid, RHO_SOOT):
        self.meanParticuleVolume = grid.meanParticuleVolume  # volumes des bins
        self.meanParticuleSurface = grid.meanParticuleSurface

        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)  # kg/m³

        # Conversion volume → masse pour les bins
        # mBins[k] = volume[k] * RHO_SOOT
        self.mBins = self.meanParticuleVolume * self.RHO_SOOT  # kg

        # Taille des sections (différence de masse entre bins consécutifs)
        # sectionSize[k] = mBins[k] - mBins[k-1] pour k=1..nsoot-1
        self.sectionSize = self.mBins[1:] - self.mBins[:-1]  # kg, shape: [nsoot-1]

    @abstractmethod
    def oxidationRate(self, ns, ys, rhos, Ts):
        """
        :param ns: concentrations en nombre de particules [batch, nsoot]
        :param ys: fractions massiques des espèces gazeuses
        :param rhos: densités
        :param Ts: températures
        :return: kOxi - taux d'oxydation en m³/s
        """
        raise NotImplementedError(
            'Any soot oxidation class inheriting from DummyOxidation must implement an oxidationRate method')

    @abstractmethod
    def getStoechiometricCoefs(self):
        """
        :return: the stoechiometric coefficient of the oxidation reaction (for consuming gaseous reactants)
        """
        raise NotImplementedError(
            'Any soot oxidation class inheriting from DummyOxidation must implement a getStoechiometricCoefs method.')

    def dndt(self, ns, ys, rhos, Ts):
        """
        Calcul du terme source d'oxydation Soxi équivalent au code C

        :param ns: [batch, nsoot] - concentrations en nombre de particules (#/m³)
        :return: dndt [batch, nsoot] - terme source (#/m³/s)
        """
        # Initialisation
        dndt = torch.zeros_like(ns)  # [batch, nsoot]

        # Surface totale par unité de volume pour chaque classe
        # Am2m3[k] = meanParticuleSurface[k] * ns[k]  (m²/m³)
        Am2m3 = self.meanParticuleSurface[None, :] * ns  # [batch, nsoot]

        # Taux d'oxydation volumétrique (m³/s)
        kOxi = self.oxidationRate(ns, ys, rhos, Ts)  # [batch, 1] ou scalaire

        # Calcul vectorisé des termes de flux (oxydation va de k vers k-1)
        # term[k] = kOxi_mass * Am2m3[k] / sectionSize[k-1] pour k=1..nsoot-1
        # Note: sectionSize[k-1] = mBins[k] - mBins[k-1]
        term = kOxi[:, None] * Am2m3[:, 1:] / self.sectionSize[None, :]  # [batch, nsoot-1]

        # Dernière classe (k=nsoot-1) : sortie uniquement
        dndt[:, -1] = -term[:, -1]

        # Classes intermédiaires (k=nsoot-2..1) : entrée depuis k+1, sortie vers k-1
        if ns.shape[1] > 2:  # s'il y a des bins intermédiaires
            dndt[:, 1:-1] = term[:, 1:] - term[:, :-1]

        # Première classe (k=0) :
        # - entrée depuis k+1 : term[0]
        # - destruction interne : -kOxi_mass * Am2m3[0] / mBins[0]
        dndt[:, 0] = term[:, 0] - kOxi * Am2m3[:, 0] / self.mBins[0]

        return dndt