import json
import os
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np
from scipy.interpolate import interp1d
import NeuralCFD.utilities_functions.primitive_converter as pc


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

    def animateFields(self, fields, variable, save_gif=False):
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
            graphs.append(ax.plot(data['x'], data[variable][0], label=os.path.splitext(os.path.split(fields[n])[1])[0])[0])
            mins.append(min([min(i) for i in data[variable]]))
            maxs.append(max([max(i) for i in data[variable]]))
        plt.legend()
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

        anim = FuncAnimation(fig, update, frames=len(simData[0]['t']), interval=50)
        if save_gif == True:
            # To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=15,
                                             metadata=dict(artist='Me'),
                                             bitrate=1800)
            anim.save(f'{variable}.gif', writer=writer)
        plt.show() #TODO: remove what is under
        #plt.show(block=False)
        #input("Animation terminée — appuie sur Entrée pour continuer...")
        #plt.close(fig)

    def computeConserved(self, thermo):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)
        rhou = []
        rhoE = []
        rhoYs = []
        for n in range(len(self.simData['t'])):
            rho = torch.tensor(self.simData['rho'][n], device=thermo.device, dtype=thermo.dtype)
            u = torch.tensor(self.simData['u'][n], device=thermo.device, dtype=thermo.dtype)
            p = torch.tensor(self.simData['P'][n], device=thermo.device, dtype=thermo.dtype)
            T = torch.tensor(self.simData['T'][n], device=thermo.device, dtype=thermo.dtype)
            Ys = []
            spe = []
            for k, v in self.simData.items():
                if 'Y_' in k:
                    Ys.append(torch.tensor(v[n], device=thermo.device, dtype=thermo.dtype))
                    spe.append(k)
            Ys = torch.stack(Ys).T
            U0 = pc.primitive_to_conserved(rho, u, p, T, Ys, thermo).cpu()
            rhou.append(U0[1, :])
            rhoE.append(U0[2, :])
            rhoYs.append(U0[3:, :])

        rhoYs = torch.stack(rhoYs, dim=-1)
        self.simData['rhou'] = rhou
        self.simData['rhoE'] = rhoE
        for i, s in enumerate(spe):
            self.simData['rho'+s] = rhoYs[i, :, :].T

    def conservationPlot(self, var):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)
        arr = np.array(self.simData[var])
        arr = np.sum(arr, axis=1)
        plt.plot(self.simData['t'], arr)
        plt.xlabel('t')
        plt.ylabel(var)
        plt.show()


    def animateField(self, variable, save_gif=False):
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

        anim = FuncAnimation(fig, update, frames=len(self.simData['t']), interval=20)
        if save_gif == True:
            # To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=15,
                                             metadata=dict(artist='Me'),
                                             bitrate=1800)
            anim.save(os.path.join(self.dataPath, f'{variable}.gif'), writer=writer)
        plt.show() #TODO: remove what is under
        #plt.show(block=False)
        #input("Animation terminée — appuie sur Entrée pour continuer...")
        #plt.close(fig)


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
