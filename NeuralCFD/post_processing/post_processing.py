import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.interpolate import interp1d

#TODO: plot (animated or not) a subpolt with rho, T, P, u and the geometry with a colormap plot to show dx

class PostProcessing:

    def __init__(self, dataPath, caseName): #TODO: make data path optional and add error messages if no data
        self.dataPath = dataPath
        self.caseName = caseName

        self.simData = None


    def __loadData(self, dataPath, caseName):
        with open(os.path.join(dataPath, caseName), 'r') as f:
            simData = json.load(f)
        return(simData)

    def animateFields(self, fields, variable):
        simData = []
        graphs = []
        mins = []
        maxs = []
        for f in fields:
            #TODO: interpolate fields on simData[0] time
            simData.append(self.__loadData('', f))
        time = simData[0]['t'] #TODO: extrapolate on the simulation with the lowest time interval
        for data in simData:
            data[variable] = interp1d(data['t'], data[variable], axis=0, bounds_error=True)(time)

        fig, ax = plt.subplots()
        for n, data in enumerate(simData):
            graphs.append(ax.plot(data['x'], data[variable][0], label=fields[n])[0])
            mins.append(min([min(i) for i in data[variable]]))
            maxs.append(max([max(i) for i in data[variable]]))
        fig.legend()
        min_y = min(mins)
        max_y = max(maxs)
        delta_y = max_y - min_y
        blank_proportion = 0.05
        plt.ylim(min_y - blank_proportion * delta_y, max_y + blank_proportion * delta_y)
        time = simData[0]['t'][0]
        plt.title(variable + f' | t = {time}')
        def update(frame):
            # creating a new graph or updating the graph
            for g, data in zip(graphs, simData):
                g.set_ydata(data[variable][frame])
            time = simData[0]['t'][frame]
            plt.title(variable + f' | t = {time:.6e}')

        anim = FuncAnimation(fig, update, frames=len(simData[0]['t']), interval=200)
        #plt.show() #TODO: remove what is under
        plt.show(block=False)
        input("Animation terminée — appuie sur Entrée pour continuer...")
        plt.close(fig)


    def animateField(self, variable):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        fig, ax = plt.subplots()
        graph = ax.plot(self.simData['x'], self.simData[variable][0])[0]
        min_y = min([min(i) for i in self.simData[variable]])
        max_y = max([max(i) for i in self.simData[variable]])
        delta_y = max_y - min_y
        blank_proportion = 0.05
        plt.ylim(min_y - blank_proportion * delta_y, max_y + blank_proportion * delta_y)
        time = self.simData['t'][0]
        plt.title(variable + f' | t = {time}')
        def update(frame):
            # creating a new graph or updating the graph
            graph.set_ydata(self.simData[variable][frame])
            time = self.simData['t'][frame]
            plt.title(variable + f' | t = {time:.6e}')

        anim = FuncAnimation(fig, update, frames=len(self.simData['t']), interval=10)
        #plt.show() #TODO: remove what is under
        plt.show(block=False)
        input("Animation terminée — appuie sur Entrée pour continuer...")
        plt.close(fig)


    def plotVariables(self, variables, cellidx): # TODO: divide in 2 function : one for ploting time slices and the other one for plotting x slices
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)
        if isinstance(variables, str) or (isinstance(variables, list) and len(variables) == 1):
            fig, ax = plt.subplots()
            axs = [ax]
        else:
            fig, axs = plt.subplots(nrows=len(variables), ncols=1)
        for n, var in enumerate(variables):
            data = np.array(self.simData[var])
            axs[n].plot(self.simData['t'], data[:, cellidx])
            axs[n].set_xlabel('t')
            axs[n].set_ylabel(var)

        plt.show()

    def compareFieldsEvolution(self, fields, variable, cellidx):
        for f in fields:
            fdata = self.__loadData(self.dataPath, f)
            var = np.array(fdata[variable])
            plt.plot(fdata['t'], var[:, cellidx], label=f)
        plt.legend()
        plt.xlabel('t')
        plt.ylabel(variable)
        plt.show()

    def plotSnapchot(self, variable, timeidx):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)
        var = np.array(self.simData[variable])
        plt.plot(self.simData['x'], var[timeidx, :])
        plt.xlabel('x')
        plt.ylabel(variable)
        plt.show()
