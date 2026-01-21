from abc import ABC, abstractmethod
import os
import json

class DummySolver(ABC):

    def __init__(self, grid, thermo, boundaryConditions, initialConditions, savingInterval, savingPath=None, initialGridTemperature=None):
        self.grid = grid
        self.thermo = thermo

        self.t = 0
        self.step = 0

        self.savingInterval = savingInterval

        self.boundaryConditions = boundaryConditions
        self.initialConditions = initialConditions
        self.initialGridTemperature = initialGridTemperature

        self.history = {'t': [], 'T': [], 'rho': [], 'P': [], 'u': []}
        for spe in self.thermo.thermoProp.species:
            self.history[f'Y_{spe}'] = []
        for i in range(self.grid.nclasses):
            self.history[f'N_{i}'] = []

        self.savingPath = savingPath


    def setIntialConditions(self):
        if self.initialGridTemperature is not None:
            self.grid.cells.T = self.initialGridTemperature
            self.grid.updateCellsState(self.initialConditions, self.boundaryConditions, usePrevT=True)
        else:
            self.grid.updateCellsState(self.initialConditions, self.boundaryConditions, usePrevT=None) # delet initial conditions to gain disk space ?

    def save(self):
        self.history['t'].append(self.t)
        self.history['T'].append(list(self.grid.cells.T.detach().clone().cpu().numpy()))
        self.history['rho'].append(list(self.grid.cells.rho.detach().clone().cpu().numpy()))
        for i, spe in enumerate(self.thermo.thermoProp.species):
            self.history[f'Y_{spe}'].append(list(self.grid.cells.Ys[:, i].detach().clone().cpu().numpy()))
        for i in range(self.grid.nclasses):
            self.history[f'N_{i}'].append(list(self.grid.cells.ns[:, i].detach().clone().cpu().numpy()))
        self.history['P'].append(list(self.grid.cells.p.detach().clone().cpu().numpy()))
        self.history['u'].append(list(self.grid.cells.u.detach().clone().cpu().numpy()))

    def addStaticSave(self):
        self.history['x'] = list(self.grid.x.detach().clone().cpu().numpy())

    def writeResults(self, caseName):
        if self.savingPath is not None:
            self.addStaticSave()
            if caseName is None:
                caseName = 'results_default_sim.json'
            if not os.path.exists(self.savingPath):
                os.makedirs(self.savingPath)
            with open(os.path.join(self.savingPath, caseName), 'w') as f:
                json.dump(self.history, f)


    @abstractmethod
    def oneStep(self, dt):
        raise NotImplementedError('A custom solver must implement oneStep(dt) methode')

    @abstractmethod
    def solve(self, tmax, maxiter, CFL, caseName=None):
        raise NotImplementedError('A custom solver must implement solve(dtMax, CFL, maxiter) methode')
