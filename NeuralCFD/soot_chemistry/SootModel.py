import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
from NeuralCFD.numerical_integrators.RK45 import RK45Integrator
import torch

#TODO: take into account the section/dx variation. (if we have a moving mesh, it will appear as a separate strang splitting operator or in the transport methodes, but no change to the chemistry solver)
class SootModel:

    RHO_SOOT = 1800.0

    def __init__(self, nucleationModel, growthModel, oxidationModel, coagulationModel, grid, thermo):
        self.nucleationModel = nucleationModel(grid, thermo, SootModel.RHO_SOOT)
        self.growthModel = growthModel(grid, thermo, SootModel.RHO_SOOT)
        self.oxidationModel = oxidationModel(grid, thermo, SootModel.RHO_SOOT)
        self.coagulationModel = coagulationModel(grid, SootModel.RHO_SOOT)

        # Intégrateur réutilisable
        self.integrator = RK45Integrator(
            lambda ns, t: ns * 0.0,  # dummy, sera remplacé
            device=grid.device,
            dtype=grid.dtype,
            verbose=False
        )

    def computeIntegratedNs(self, grid, dt):
        """
        Intégration RK45 du système de suie

        Équation bilan:
        dns/dt = (nucléation) + (croissance) + (oxydation) + (coagulation)
        """
        # To be included in state vector when retroaction will be added
        ys = grid.cells.Ys
        rhos = grid.cells.rho
        Ts = grid.cells.T
        def rhs_soot(ns, t):
            """
            Right-hand side pour équations de population
            """
            ns = torch.clamp(ns, min=0.0)

            dn_nucleation = self.nucleationModel.dndt(ns, ys, rhos, Ts)
            dn_growth = self.growthModel.dndt(ns, ys, rhos, Ts)
            dn_oxyd = self.oxidationModel.dndt(ns, ys, rhos, Ts)
            dn_coag = self.coagulationModel.dndt(ns, ys, rhos, Ts)

            return dn_nucleation + dn_growth + dn_oxyd + dn_coag

        # Intégration RK45
        self.integrator.f = rhs_soot


        ns0 = grid.cells.ns.clone()
        t_vals, ns_vals = self.integrator.integrate(ns0, t_span=(0.0, dt))
        ns_final = ns_vals[-1]

        ns_total = torch.clamp(ns_final, min=0.0)

        return ns_total

    def soot(self, dt, grid, thermo, bc):
        U_new = primitive_converter.primitive_to_conserved(grid.cells.rho, grid.cells.u, grid.cells.p, grid.cells.T, grid.cells.Ys, self.computeIntegratedNs(grid, dt), thermo)
        grid.updateCellsState(U_new, bc)