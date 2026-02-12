import torch
from abc import ABC, abstractmethod

#TODO: take into account the section/dx variation. (if we have a mooving mesh, it will appear as a separate strang splitting operator or in the transport methodes, but no change to the chemistry solver)
class DummyNucleation(ABC):
    def __init__(self, grid, RHO_SOOT):
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)  # kg/m³ (densité suie)
        self.MW_C = torch.tensor(12e-3, device=grid.device, dtype=grid.dtype)  # (kg/mol)
        self.Na = torch.tensor(6.022e23, device=grid.device, dtype=grid.dtype)  # #/mol
        # ============== PARAMÈTRES DE NUCLÉATION ==============

        # Volume de la particule nucléée (m³)
        # self.v_nuc = (π/6) × d³
        self.v_nuc = grid.meanParticuleVolume[0]

        # Masse de la particule nucléée (kg)
        self.m_nuc = self.v_nuc * self.RHO_SOOT

        # Masse d'une particule dans une section donnée (kg)
        self.m_class = grid.meanParticuleVolume * self.RHO_SOOT

        # Nb de carbonnes de la particule nucléée
        # m_nuc = C_min × MW_C / Na
        self.C_min = self.m_nuc * self.Na / self.MW_C

    @abstractmethod
    def nucleationRate(self, ns, ys, rhos, Ts):
        """
        :param grid: the computation grid as defined in NeuralCFD.grid.grid
        :return: The nucleation rate in #/m3/s for the smallest class
        """
        raise NotImplementedError('Any soot nucleation class inheriting from DummyNucleation must implement a nucleationRate method')

    @abstractmethod
    def getStoechiometricCoefs(self):
        """
        :return: the stoechiometric coefficient of the nucleation reaction (for consuming gaseous reactants)
        """
        raise NotImplementedError('Any soot nucleation class inheriting from DummyNucleation must implement a getStoechiometricCoefs method.')

    def dndt(self, ns, ys, rhos, Ts):
        # Créer les particules dans la classe correspondant à v_nuc
        dndt = torch.zeros_like(ns)

        S_nuc_0 = self.nucleationRate(ns, ys, rhos, Ts) * self.m_nuc / self.m_class[0]  # It should conserve mass if there is a slight missalignment between m_nuc and m_class

        # Ajouter les particules nucléées à cette classe
        dndt[:, 0] = S_nuc_0

        # Ajouter le calcul des dérivées de toutes les autres grandeurs pour un couplage complet
        pass

        return dndt