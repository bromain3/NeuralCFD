import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
from abc import ABC, abstractmethod

#TODO: take into account the section/dx variation. (if we have a mooving mesh, it will appear as a separate strang splitting operator or in the transport methodes, but no change to the chemistry solver)
class DummySoot(ABC):

    @abstractmethod
    def computeIntegratedNs(self, grid, dt):
       raise NotImplementedError('Any soot class inheriting from DummySoot must implement a blabla method.')


    def soot(self, dt, grid, thermo, bc):
        #dndt = self.computeSources(grid, dt)
        #U_new = primitive_converter.primitive_to_conserved(grid.cells.rho, grid.cells.u, grid.cells.p, grid.cells.T, grid.cells.Ys, grid.cells.ns + dt * dndt, thermo)
        U_new = primitive_converter.primitive_to_conserved(grid.cells.rho, grid.cells.u, grid.cells.p, grid.cells.T,
                                                           grid.cells.Ys, self.computeIntegratedNs(grid, dt), thermo)
        grid.updateCellsState(U_new, bc)