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
        Calcul du terme source d'oxydation Soxi équivalent au code C++ SECT
        L'oxydation crée une vélocité négative dans le domaine de taille (les particules rétrécissent)

        :param ns: [batch, nsoot] - concentrations en nombre de particules (#/m³)
        :param ys: [batch, n_species] - fractions massiques des espèces gazeuses
        :param rhos: [batch] - densité du mélange gazeux (kg/m³)
        :param Ts: [batch] - température (K)
        :return: dndt [batch, nsoot] - terme source (#/m³/s)
        """
        nsoot = ns.shape[1]

        # Initialisation
        dndt = torch.zeros_like(ns)  # [batch, nsoot]

        # Taux d'oxydation massique (kg/m²/s)
        kOxi = self.oxidationRate(ns, ys, rhos, Ts)  # [batch]

        # Surface totale par unité de volume pour chaque classe
        # Am2m3[k] = meanParticuleSurface[k] * ns[k]  (m²/m³)
        Am2m3 = self.meanParticuleSurface[None, :] * ns  # [batch, nsoot]

        # k=nsoot-1: dernière classe (sortie uniquement)
        k = nsoot - 1
        term = kOxi * Am2m3[:, k] / (self.mBins[k] - self.mBins[k - 1])  # [batch]
        dndt[:, k] = -term

        # k=nsoot-2 à k=1: classes intermédiaires (entrée depuis k+1, sortie vers k-1)
        # Boucle descendante de nsoot-2 vers 1
        for k in range(nsoot - 2, 0, -1):
            dndt[:, k] = term  # entrée depuis k+1
            term = kOxi * Am2m3[:, k] / (self.mBins[k] - self.mBins[k - 1])  # [batch]
            dndt[:, k] -= term  # sortie vers k-1

        # k=0: première classe (entrée depuis k+1 - destruction interne)
        k = 0
        dndt[:, k] = term - kOxi * Am2m3[:, k] / self.mBins[k]

        return dndt