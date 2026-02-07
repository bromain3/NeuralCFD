import json
import os
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np
from scipy.interpolate import interp1d
import NeuralCFD.utilities_functions.primitive_converter as pc
from matplotlib import cm


# TODO: plot (animated or not) a subpolt with rho, T, P, u and the geometry with a colormap plot to show dx

class PostProcessing:

    def __init__(self, dataPath, caseName):  # TODO: make data path optional and add error messages if no data
        self.dataPath = dataPath
        self.caseName = caseName

        self.simData = None

    def __loadData(self, dataPath, caseName):
        with open(os.path.join(dataPath, caseName), 'r') as f:
            simData = json.load(f)
        return (simData)

    def animateFields(self, fields, variable, save_gif=False):
        simData = []
        graphs = []
        mins = []
        maxs = []
        for f in fields:
            # TODO: interpolate fields on simData[0] time
            simData.append(self.__loadData('', f))
        time = simData[0]['t']  # TODO: extrapolate on the simulation with the lowest time interval
        for data in simData:
            data[variable] = interp1d(data['t'], data[variable], axis=0, bounds_error=True)(time)

        fig, ax = plt.subplots()
        for n, data in enumerate(simData):
            graphs.append(
                ax.plot(data['x'], data[variable][0], label=os.path.splitext(os.path.split(fields[n])[1])[0])[0])
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
        plt.show()  # TODO: remove what is under
        # plt.show(block=False)
        # input("Animation terminée — appuie sur Entrée pour continuer...")
        # plt.close(fig)

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
            self.simData['rho' + s] = rhoYs[i, :, :].T

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
        plt.show()  # TODO: remove what is under
        # plt.show(block=False)
        # input("Animation terminée — appuie sur Entrée pour continuer...")
        # plt.close(fig)

    def plotSpatialSootEvolution3D(self, time_idx, log_scale=True, y_axis='class'):
        """
        Affiche l'évolution spatiale des suies en 3D

        Parameters:
        -----------
        time_idx : int
            Indice temporel
        log_scale : bool
            Utiliser une échelle logarithmique
        y_axis : str
            'class' : afficher le numéro de classe
            'size' : afficher la taille moyenne des suies (nm³)
            'log_size' : afficher log10 de la taille moyenne des suies (nm³)
        """
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        x_coords = self.simData['x']
        class_edges_nm3 = np.array(self.simData['class_edges']) * 1e27
        section_size_nm3 = np.array(self.simData['section_size']) * 1e27
        n_classes = len(class_edges_nm3) - 1

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Choisir les valeurs de l'axe Y selon l'option
        if y_axis == 'class':
            y_values = np.arange(n_classes)
            ylabel = 'Classe de taille'
        elif y_axis == 'size':
            y_values = section_size_nm3
            ylabel = 'Taille moyenne des suies (nm³)'
        elif y_axis == 'log_size':
            y_values = np.log10(section_size_nm3)
            ylabel = 'log10(Taille moyenne des suies) (nm³)'
        else:
            raise ValueError("y_axis doit être 'class', 'size' ou 'log_size'")

        # Créer des arrays pour le maillage
        X, Y = np.meshgrid(x_coords, y_values)
        Z = np.zeros_like(X, dtype=float)

        # Remplir Z avec les concentrations
        for i in range(n_classes):
            Ni_spatial = np.array(self.simData[f'N_{i}'])[time_idx, :]
            if log_scale:
                Z[i, :] = np.maximum(np.log10(Ni_spatial + 1e-300), 0.0)
            else:
                Z[i, :] = Ni_spatial

        # Tracer la surface
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')

        ax.set_xlabel('Position x (m)')
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_zlabel('log10(Concentration Ni [#/m3])')
        else:
            ax.set_zlabel('Concentration Ni [#/m3]')

        ax.set_title(f'Distribution Spatiale des Suies à t = {self.simData["t"][time_idx]:.4e} s')

        # Ajouter une colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()

    def plotTemporalSootEvolution3D(self, cell_idx, log_scale=True, y_axis='class'):
        """
        Affiche l'évolution temporelle des suies en 3D

        Parameters:
        -----------
        cell_idx : int
            Indice spatial
        log_scale : bool
            Utiliser une échelle logarithmique
        y_axis : str
            'class' : afficher le numéro de classe
            'size' : afficher la taille moyenne des suies (nm³)
            'log_size' : afficher log10 de la taille moyenne des suies (nm³)
        """
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        time_coords = self.simData['t']
        class_edges_nm3 = np.array(self.simData['class_edges']) * 1e27
        section_size_nm3 = np.array(self.simData['section_size']) * 1e27
        n_classes = len(class_edges_nm3) - 1

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Choisir les valeurs de l'axe Y selon l'option
        if y_axis == 'class':
            y_values = np.arange(n_classes)
            ylabel = 'Classe de taille'
        elif y_axis == 'size':
            y_values = section_size_nm3
            ylabel = 'Taille moyenne des suies (nm³)'
        elif y_axis == 'log_size':
            y_values = np.log10(section_size_nm3)
            ylabel = 'log10(Taille moyenne des suies) (nm³)'
        else:
            raise ValueError("y_axis doit être 'class', 'size' ou 'log_size'")

        # Créer des arrays pour le maillage
        T, Y = np.meshgrid(time_coords, y_values)
        Z = np.zeros_like(T, dtype=float)

        # Remplir Z avec les concentrations
        for i in range(n_classes):
            Ni_temporal = np.array(self.simData[f'N_{i}'])[:, cell_idx]
            if log_scale:
                Z[i, :] = np.maximum(np.log10(Ni_temporal + 1e-300), 0.0)
            else:
                Z[i, :] = Ni_temporal

        # Tracer la surface
        surf = ax.plot_surface(T, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')

        ax.set_xlabel('Temps (s)')
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_zlabel('log10(Concentration Ni [#/m3])')
        else:
            ax.set_zlabel('Concentration Ni [#/m3])')

        ax.set_title(f'Évolution Temporelle des Suies à x = {self.simData["x"][cell_idx]:.4e} m')

        # Ajouter une colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()

    def plotSpatialSootEvolutionAnimated3D(self, log_scale=True, y_axis='class', interval=100, save_path=None):
        """
        Anime l'évolution spatiale des suies en 3D au cours du temps

        Parameters:
        -----------
        log_scale : bool
            Utiliser une échelle logarithmique
        y_axis : str
            'class' : afficher le numéro de classe
            'size' : afficher la taille moyenne des suies (nm³)
            'log_size' : afficher log10 de la taille moyenne des suies (nm³)
        interval : int
            Délai entre les frames en millisecondes
        save_path : str, optional
            Chemin pour sauvegarder l'animation (ex: 'animation.gif' ou 'animation.mp4')
        """
        from matplotlib.animation import FuncAnimation

        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        x_coords = self.simData['x']
        time_coords = self.simData['t']
        class_edges_nm3 = np.array(self.simData['class_edges']) * 1e27
        section_size_nm3 = np.array(self.simData['section_size']) * 1e27
        n_classes = len(class_edges_nm3) - 1

        # Choisir les valeurs de l'axe Y selon l'option
        if y_axis == 'class':
            y_values = np.arange(n_classes)
            ylabel = 'Classe de taille'
        elif y_axis == 'size':
            y_values = section_size_nm3
            ylabel = 'Taille moyenne des suies (nm³)'
        elif y_axis == 'log_size':
            y_values = np.log10(section_size_nm3)
            ylabel = 'log10(Taille moyenne des suies) (nm³)'
        else:
            raise ValueError("y_axis doit être 'class', 'size' ou 'log_size'")

        # Préparer la figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Créer le maillage (constant)
        X, Y = np.meshgrid(x_coords, y_values)

        # Calculer les limites Z pour toute l'animation (pour garder les axes fixes)
        Z_min, Z_max = np.inf, -np.inf
        for time_idx in range(len(time_coords)):
            for i in range(n_classes):
                Ni_spatial = np.array(self.simData[f'N_{i}'])[time_idx, :]
                if log_scale:
                    values = np.maximum(np.log10(Ni_spatial + 1e-300), 0.0)
                else:
                    values = Ni_spatial
                Z_min = min(Z_min, np.min(values))
                Z_max = max(Z_max, np.max(values))

        # Initialiser la surface
        Z = np.zeros_like(X, dtype=float)
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none',
                               vmin=Z_min, vmax=Z_max)

        # Configuration des axes
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_zlabel('log10(Concentration Ni [#/m3])')
        else:
            ax.set_zlabel('Concentration Ni [#/m3]')
        ax.set_zlim(Z_min, Z_max)

        title = ax.set_title(f'Distribution Spatiale des Suies à t = {time_coords[0]:.4e} s')

        # Ajouter une colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        def update(frame):
            ax.clear()

            # Remplir Z avec les nouvelles données
            Z = np.zeros_like(X, dtype=float)
            for i in range(n_classes):
                Ni_spatial = np.array(self.simData[f'N_{i}'])[frame, :]
                if log_scale:
                    Z[i, :] = np.maximum(np.log10(Ni_spatial + 1e-300), 0.0)
                else:
                    Z[i, :] = Ni_spatial

            # Tracer la nouvelle surface
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none',
                                   vmin=Z_min, vmax=Z_max)

            # Reconfigurer les axes
            ax.set_xlabel('Position x (m)')
            ax.set_ylabel(ylabel)
            if log_scale:
                ax.set_zlabel('log10(Concentration Ni [#/m3])')
            else:
                ax.set_zlabel('Concentration Ni [#/m3]')
            ax.set_zlim(Z_min, Z_max)
            ax.set_title(f'Distribution Spatiale des Suies à t = {time_coords[frame]:.4e} s')

            return surf,

        # Créer l'animation
        anim = FuncAnimation(fig, update, frames=len(time_coords),
                             interval=interval, blit=False, repeat=True)

        # Sauvegarder si demandé
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000 / interval)
                print(f"Animation sauvegardée : {save_path}")
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000 / interval)
                print(f"Animation sauvegardée : {save_path}")
            else:
                print("Format non supporté. Utilisez .gif ou .mp4")

        plt.tight_layout()
        plt.show()

        return anim

    def plotSpatialSootEvolution2D(self, time_idx, log_scale=True):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        x_coords = self.simData['x']

        # Utiliser les bords de classes pour l'axe Y
        # Convertir en nm si ce n'est pas déjà fait
        class_edges_nm3 = np.array(self.simData['class_edges']) * 1e27

        fig, ax = plt.subplots(figsize=(10, 6))

        # Pour chaque classe, tracer l'évolution spatiale
        for i in range(len(class_edges_nm3) - 1):  # -1 car class_edges a un élément de plus que le nombre de classes
            Ni_spatial = np.array(self.simData[f'N_{i}'])[time_idx, :]

            # Utiliser la borne supérieure de la classe pour le label, ou le volume moyen
            label_text = f'Class {i} ({class_edges_nm3[i]:.2f}-{class_edges_nm3[i + 1]:.2f} nm3)'

            if log_scale:
                ax.plot(x_coords, np.log10(Ni_spatial + 1e-300), label=label_text)  # Ajout de 1e-15 pour éviter log(0)
                ax.set_ylabel('log10(Concentration Ni [#/m3])')
            else:
                ax.plot(x_coords, Ni_spatial, label=label_text)
                ax.set_ylabel('Concentration Ni [#/m3]')

        ax.set_xlabel('Position x (m)')
        ax.set_title(f'Distribution Spatiale des Suies à t = {self.simData["t"][time_idx]:.4e} s')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plotTemporalSootEvolution2D(self, cell_idx, log_scale=True):
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        time_coords = self.simData['t']

        # Utiliser les bords de classes pour l'axe Y
        # Convertir en nm si ce n'est pas déjà fait
        class_edges_nm3 = np.array(self.simData['class_edges']) * 1e27

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(class_edges_nm3) - 1):
            Ni_temporal = np.array(self.simData[f'N_{i}'])[:, cell_idx]
            label_text = f'Class {i} ({class_edges_nm3[i]:.2f}-{class_edges_nm3[i + 1]:.2f} nm3)'

            if log_scale:
                ax.plot(time_coords, np.log10(Ni_temporal + 1e-15), label=label_text)
                ax.set_ylabel('log10(Concentration Ni [#/m3])')
            else:
                ax.plot(time_coords, Ni_temporal, label=label_text)
                ax.set_ylabel('Concentration Ni [#/m3]')

        ax.set_xlabel('Temps (s)')
        ax.set_title(f'Évolution Temporelle des Suies à x = {self.simData["x"][cell_idx]:.4e} m')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plotVariables(self, variables,
                      cellidx):  # TODO: divide in 2 function : one for ploting time slices and the other one for plotting x slices
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

    def plotSootVolumeFraction(self, timeidx, log_scale=False, unit='ppm'):
        """
        Trace la fraction volumique de suies pour un instant de temps donné

        Parameters:
        -----------
        timeidx : int
            Indice temporel
        log_scale : bool
            Utiliser une échelle logarithmique pour l'axe y
        unit : str
            'fraction' : afficher en fraction volumique
            'ppm' : afficher en parties par million
        """
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        x_coords = self.simData['x']
        section_size_m3 = np.array(self.simData['mean_particule_volume'])  # Volume moyen par classe en m³
        n_classes = len(section_size_m3)

        # Calculer la fraction volumique totale
        fv_total = np.zeros(len(x_coords))

        for i in range(n_classes):
            Ni_spatial = np.array(self.simData[f'N_{i}'])[timeidx, :]  # Concentration en #/m³
            fv_i = Ni_spatial * section_size_m3[i]  # Fraction volumique de la classe i
            fv_total += fv_i

        # Conversion en ppm si demandé
        if unit == 'ppm':
            fv_total = fv_total * 1e6
            ylabel = 'Fraction volumique de suies (ppm)'
        else:
            ylabel = 'Fraction volumique de suies'

        fig, ax = plt.subplots(figsize=(10, 6))

        if log_scale:
            ax.semilogy(x_coords, fv_total, 'b-', linewidth=2)
            ylabel += ' (log scale)'
        else:
            ax.plot(x_coords, fv_total, 'b-', linewidth=2)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Position x (m)')
        ax.set_title(f'Fraction volumique de suies à t = {self.simData["t"][timeidx]:.4e} s')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def animateSootVolumeFraction(self, log_scale=False, save_gif=False, unit='ppm'):
        """
        Anime l'évolution de la fraction volumique de suies au cours du temps

        Parameters:
        -----------
        log_scale : bool
            Utiliser une échelle logarithmique pour l'axe y
        save_gif : bool
            Sauvegarder l'animation en gif
        unit : str
            'fraction' : afficher en fraction volumique
            'ppm' : afficher en parties par million
        """
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)

        x_coords = self.simData['x']
        time_coords = self.simData['t']
        section_size_m3 = np.array(self.simData['mean_particule_volume'])
        n_classes = len(section_size_m3)

        # Pré-calculer toutes les fractions volumiques
        fv_all_times = []
        for time_idx in range(len(time_coords)):
            fv_total = np.zeros(len(x_coords))
            for i in range(n_classes):
                Ni_spatial = np.array(self.simData[f'N_{i}'])[time_idx, :]
                fv_i = Ni_spatial * section_size_m3[i]
                fv_total += fv_i
            fv_all_times.append(fv_total)

        fv_all_times = np.array(fv_all_times)

        # Conversion en ppm si demandé
        if unit == 'ppm':
            fv_all_times = fv_all_times * 1e6
            ylabel = 'Fraction volumique de suies (ppm)'
        else:
            ylabel = 'Fraction volumique de suies'

        # Déterminer les limites des axes
        if log_scale:
            fv_min = np.min(fv_all_times[fv_all_times > 0]) if np.any(fv_all_times > 0) else 1e-20
            fv_max = np.max(fv_all_times)
            ylabel += ' (log scale)'
        else:
            fv_min = np.min(fv_all_times)
            fv_max = np.max(fv_all_times)

        delta_y = fv_max - fv_min
        blank_proportion = 0.05

        fig, ax = plt.subplots(figsize=(10, 6))
        graph, = ax.plot(x_coords, fv_all_times[0], 'b-', linewidth=2)

        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(fv_min * 0.5, fv_max * 2)
        else:
            ax.set_ylim(fv_min - blank_proportion * delta_y,
                        fv_max + blank_proportion * delta_y)

        ax.set_xlabel('Position x (m)')
        ax.set_ylabel(ylabel)

        title = ax.set_title(f'Fraction volumique de suies | t = {time_coords[0]:.6e} s')
        ax.grid(True, alpha=0.3)

        def update(frame):
            graph.set_ydata(fv_all_times[frame])
            title.set_text(f'Fraction volumique de suies | t = {time_coords[frame]:.6e} s')
            return graph, title

        anim = FuncAnimation(fig, update, frames=len(time_coords), interval=50, blit=True)

        if save_gif:
            writer = animation.PillowWriter(fps=15,
                                            metadata=dict(artist='Me'),
                                            bitrate=1800)
            anim.save(os.path.join(self.dataPath, 'soot_volume_fraction.gif'), writer=writer)
            print(f"Animation sauvegardée : {os.path.join(self.dataPath, 'soot_volume_fraction.gif')}")

        plt.tight_layout()
        plt.show()