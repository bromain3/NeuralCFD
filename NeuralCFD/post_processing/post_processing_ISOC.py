import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import LogFormatter, ScalarFormatter
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.integrate import trapz, simps
import re
from joblib import Parallel, delayed
from tqdm import tqdm


class PostProcessing:

    def __init__(self, dataPath, caseName):  # TODO: make data path optional and add error messages if no data
        self.dataPath = dataPath
        self.caseName = caseName

        self.simData = None

    def __loadData(self, dataPath, caseName):
        with open(os.path.join(dataPath, caseName), 'r') as f:
            simData = json.load(f)
        return (simData)

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

        anim = FuncAnimation(fig, update, frames=len(self.simData['t']), interval=40)
        plt.show()

    def plotVariable(self, variable,
                     cellidx):  # TODO: divide in 2 function : one for ploting time slices and the other one for plotting x slices
        if self.simData is None:
            self.simData = self.__loadData(self.dataPath, self.caseName)
        var = np.array(self.simData[variable])
        plt.plot(self.simData['t'], var[:, cellidx])
        plt.xlabel('t')
        plt.ylabel(variable)
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

    def compareFieldsEvolutionISOC(self, fields, variable, cellidx):

        label_conversion = {
            'auto_allumage_full_cantera.json': 'baseline',
            'auto_allumage_full_neural.json': 'full NN',
            'auto_allumage_reduced_neural.json': 'NN - NN thermo',
            'auto_allumage_reduced_neural_approx_thermo.json': 'NASA7 thermo',
            'auto_allumage_DRGEP_35_cantera.json': 'DRGEP 35',
            'auto_allumage_DRGEP_24_cantera.json': 'DRGEP 24',
            'auto_allumage_SAR_30_cantera.json': 'SARGEP 30',
            'auto_allumage_SAR_28_cantera.json': 'SARGEP 28',
            'auto_allumage_SAR_15_cantera.json': 'SARGEP 15',
        }

        plt.figure()
        ax = plt.gca()
        # === Création de l'inset ===
        # Position et taille de l'encart [x0, y0, largeur, hauteur] en fraction de la figure
        axins = ax.inset_axes((0.00014 - 0.0003, 0.0003, 0.0003, 0.0025), transform=ax.transData)

        for f in fields:
            fdata = self.__loadData(self.dataPath, f)
            var = np.array(fdata[variable])
            ax.plot(fdata['t'], var[:, cellidx], label=label_conversion[f])
            # Tracé des mêmes données dans l'encart
            axins.plot(fdata['t'], var[:, cellidx])

        ax.legend(loc=(0.45, 0.16))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('$Y_{C_2H_2}$ [---]')

        # Limites du zoom (par exemple autour du pic principal)
        x1, x2 = 0.000206, 0.0002275
        y1, y2 = 0, 0.00305
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)

        ax.set_xlim(-0.0002, 0.00095)  # -6e-5, 0.001313

        # Ajouter un rectangle sur le plot principal pour indiquer la zone zoomée
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5", linestyle="--", linewidth=0.5)

        plt.tight_layout(pad=0.09)
        plt.show()

    def ISOCStats(self, fields, cellidx=0, baseline=None):
        from scipy.integrate import trapz
        import pandas as pd
        """
        Calcule et affiche les statistiques d'ignition pour chaque champ :
          - délai d'auto-inflammation (max de dT/dt)
          - max de Y_C2H2
          - aire sous la courbe Y_C2H2(t)
        Si une baseline est fournie, calcule aussi les erreurs relatives et absolues.

        Args:
            fields (list[str]): noms des champs à traiter
            cellidx (int): index de cellule à analyser (défaut = 0)
            baseline (str | None): nom du champ de référence

        Returns:
            pd.DataFrame: tableau des statistiques avec comparaisons
        """
        results = []

        # --- Boucle sur les champs ---
        for f in fields:
            fdata = self.__loadData(self.dataPath, f)
            t = np.array(fdata['t'])
            T = np.array(fdata['T'])[:, cellidx]
            Y_C2H2 = np.array(fdata['Y_C2H2'])[:, cellidx]

            # 1️⃣ délai d’auto-inflammation (max de dT/dt)
            dTdt = np.gradient(T, t)
            t_autoign = t[np.argmax(dTdt)]

            # 2️⃣ max de Y_C2H2
            Ymax = np.max(Y_C2H2)

            # 3️⃣ aire sous la courbe
            area = trapz(Y_C2H2, t)

            results.append({
                "Field": f,
                "t_autoign [s]": t_autoign,
                "Y_C2H2_max": Ymax,
                "Y_C2H2_area": area
            })

        df = pd.DataFrame(results)
        df = df.sort_values("t_autoign [s]").reset_index(drop=True)

        # --- Comparaison à la baseline ---
        if baseline is not None:
            if baseline not in df["Field"].values:
                raise ValueError(f"Baseline '{baseline}' n'est pas dans la liste des champs.")

            base_row = df[df["Field"] == baseline].iloc[0]

            # Calcul des erreurs
            for var in ["t_autoign [s]", "Y_C2H2_max", "Y_C2H2_area"]:
                df[f"{var} abs err"] = np.abs(df[var] - base_row[var])
                df[f"{var} rel err [%]"] = 100 * np.abs((df[var] - base_row[var]) / base_row[var])

        # --- Affichage ---
        print("\n=== Résumé ISOC ===")
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:.4e}" if isinstance(x, (float, np.floating)) else str(x)))

        return df

    def compareModelToBaseline(
            self,
            model_dir: str,
            baseline_dir: str,
            variable: str = "T",
            metric: str = "MAE",
            cellidx: int = 0,
            save_fig: bool = False,
            log_scale: bool = False,
            figsize=(3.5, 2.5),
            levels=None,
            title='',
            xlabel='',
            ylabel='',
            colorbarLabel='',
            n_jobs: int = 4,
            verbose: bool = True,
    ):
        """
        Compare un modèle chimique à un baseline pour tous les fichiers T_{}_phi_{}.json.

        Args:
            model_dir (str): dossier contenant les résultats du modèle.
            baseline_dir (str): dossier contenant les résultats du modèle de référence.
            variable (str): variable à comparer (ex: 'T', 'Y_C2H2', etc.).
            metric (str): "MAE", "MAE%", "MAX", "MAX%", "AREA_ERR%".
            cellidx (int): index de cellule à comparer.
            save_fig (bool): si True, sauvegarde les figures.
            cmap (str): colormap matplotlib.
        """

        assert metric in ["MAE", "MAE%", "MAX", "MAX%",
                          "AREA_ERR%"], "Metric doit être 'MAE', 'MAE%', 'MAX', 'MAX%' ou 'AREA_ERR%'."

        # Regex pour extraire T et phi depuis le nom du fichier
        pattern = re.compile(r"T_([\d\.eE\+\-]+)_phi_([\d\.eE\+\-]+)\.json")

        results = []

        model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
        base_files = [f for f in os.listdir(baseline_dir) if f.endswith(".json")]

        # Fichiers communs (mêmes conditions)
        common_files = list(set(model_files).intersection(set(base_files)))
        if not common_files:
            raise ValueError("Aucun fichier commun trouvé entre le modèle et le baseline.")

        # --- 🔁 Fonction pour traiter un fichier ---
        def process_file(filename):
            match = pattern.match(filename)
            if not match:
                return None
            T_val, phi_val = float(match.group(1)), float(match.group(2))

            try:
                with open(os.path.join(model_dir, filename)) as f:
                    model_data = json.load(f)
                with open(os.path.join(baseline_dir, filename)) as f:
                    base_data = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Impossible de lire {filename}: {e}")
                return None

            # Données
            t_model = np.array(model_data['t'])
            t_base = np.array(base_data['t'])
            if 'concentration' in variable:
                y_model = np.array(model_data['rho'])[:, cellidx] * np.array(model_data['Y_' + variable.split('_')[1]])[
                                                                    :, cellidx]
                y_base = np.array(base_data['rho'])[:, cellidx] * np.array(base_data['Y_' + variable.split('_')[1]])[:,
                                                                  cellidx]
            else:
                y_model = np.array(model_data[variable])[:, cellidx]
                y_base = np.array(base_data[variable])[:, cellidx]

            # Interpolation du baseline sur la grille du modèle
            try:
                interp_base = interp1d(t_base, y_base, bounds_error=False, fill_value="extrapolate")
                y_base_interp = interp_base(t_model)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Interpolation échouée pour {filename}: {e}")
                return None

            # Calcul de la métrique
            if metric == "MAE":
                err = np.mean(np.abs(y_model - y_base_interp))
            elif metric == "MAE%":
                err = 100 * np.mean(np.abs((y_model - y_base_interp) / (np.abs(y_base_interp) + 1e-12)))
            elif metric == "MAX%":
                err = np.max(np.abs((y_model - y_base_interp) / (np.abs(y_base_interp) + 1e-12)))
            elif metric == "MAX":
                err = np.max(np.abs(y_model - y_base_interp))
            elif metric == "AREA_ERR%":
                area_model = trapz(y_model, t_model)
                area_base = simps(y_base, t_base)
                err = 100 * np.abs(area_model - area_base) / np.abs(area_base) if area_base != 0 else np.nan
            else:
                raise NotImplementedError('metric not implemented')

            return (T_val, phi_val, err)

            # --- ⚡️ Lancement parallèle ---

        if verbose:
            print(f"🔄 Traitement de {len(common_files)} fichiers sur {n_jobs} cœurs...")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_file)(fname) for fname in tqdm(common_files)
        )
        results = [r for r in results if r is not None]
        results = np.array(results)

        if len(results) == 0:
            raise RuntimeError("Aucun résultat valide obtenu — vérifier les fichiers.")

        # --- 📊 Mise en forme pour le plot ---
        T_vals = np.unique(results[:, 0])
        phi_vals = np.unique(results[:, 1])
        Z = np.full((len(phi_vals), len(T_vals)), np.nan)

        for T, phi, err in results:
            i = np.where(phi_vals == phi)[0][0]
            j = np.where(T_vals == T)[0][0]
            Z[i, j] = err

        # --- 🎨 Plot publication-ready ---
        fig, ax = plt.subplots(figsize=figsize)

        cmap = plt.get_cmap("coolwarm")
        norm = mpl.colors.LogNorm(vmin=np.nanmin(Z[Z > 0]), vmax=np.nanmax(Z)) if log_scale else None

        # use contourf ?

        c = ax.imshow(
            Z,
            extent=[T_vals.min(), T_vals.max(), phi_vals.min(), phi_vals.max()],
            origin='lower',
            aspect='auto',
            cmap=cmap,
            norm=norm,
            interpolation='bilinear',  # interpolation visuelle
            rasterized=True
        )
        """
        c = ax.pcolormesh(
            T_vals, phi_vals, Z,
            shading='gouraud',
            cmap=cmap,
            norm=norm,
            rasterized=True
        )
        """
        cs = ax.contour(T_vals, phi_vals, Z, colors='k', levels=levels, linewidths=0.3, alpha=0.4)
        ax.clabel(cs, inline=True, fmt="%.1f", fontsize=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        cbar = fig.colorbar(c, ax=ax, aspect=20)
        cbar.set_label(colorbarLabel)
        if log_scale:
            cbar.ax.yaxis.set_major_formatter(LogFormatter())
        else:
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.tick_params(which='both', direction='in')
        fig.tight_layout(pad=0.1)

        if save_fig:
            fname = f"error_map_{model_dir}_{variable}_{metric}"
            plt.savefig(fname)
            print(f"✅ Figure sauvegardée : {fname}")
        else:
            plt.show()

        return results

    def compareTwoModelsToBaselineISOC(
            self,
            model_dir1: str,
            model_dir2: str,
            baseline_dir: str,
            variable: str = "T",
            metric: str = "MAE",
            cellidx: int = 0,
            save_fig: bool = False,
            log_scale: bool = False,
            figsize=(7, 2.3),
            levels=None,
            title1='Model 1',
            title2='Model 2',
            xlabel='',
            ylabel='',
            colorbarLabel='',
            n_jobs: int = 4,
            verbose: bool = True,
    ):
        """
        Compare deux modèles chimiques à un baseline et affiche côte à côte les cartes d'erreur.

        Args:
            model_dir1 (str): dossier contenant les résultats du modèle 1.
            model_dir2 (str): dossier contenant les résultats du modèle 2.
            baseline_dir (str): dossier contenant les résultats du baseline.
            variable (str): variable à comparer (ex: 'T', 'Y_C2H2', etc.).
            metric (str): "MAE", "MAE%", "MAX", "MAX%", "AREA_ERR%".
            cellidx (int): index de cellule à comparer.
            save_fig (bool): si True, sauvegarde la figure.
        """
        import os, re, json
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.ticker import LogFormatter, ScalarFormatter
        from scipy.interpolate import interp1d
        from scipy.integrate import trapz, simps
        from joblib import Parallel, delayed
        from tqdm import tqdm

        assert metric in ["MAE", "MAE%", "MAX", "MAX%",
                          "AREA_ERR%"], "Metric doit être 'MAE', 'MAE%', 'MAX', 'MAX%' ou 'AREA_ERR%'."

        pattern = re.compile(r"T_([\d\.eE\+\-]+)_phi_([\d\.eE\+\-]+)\.json")

        def process_model(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
            base_files = [f for f in os.listdir(baseline_dir) if f.endswith(".json")]
            common_files = list(set(model_files).intersection(set(base_files)))
            if not common_files:
                raise ValueError(f"Aucun fichier commun trouvé entre {model_dir} et {baseline_dir}.")

            def process_file(filename):
                match = pattern.match(filename)
                if not match:
                    return None
                T_val, phi_val = float(match.group(1)), float(match.group(2))

                try:
                    with open(os.path.join(model_dir, filename)) as f:
                        model_data = json.load(f)
                    with open(os.path.join(baseline_dir, filename)) as f:
                        base_data = json.load(f)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Impossible de lire {filename}: {e}")
                    return None

                t_model = np.array(model_data['t'])
                t_base = np.array(base_data['t'])
                if 'concentration' in variable:
                    y_model = np.array(model_data['rho'])[:, cellidx] * np.array(
                        model_data['Y_' + variable.split('_')[1]])[:, cellidx]
                    y_base = np.array(base_data['rho'])[:, cellidx] * np.array(
                        base_data['Y_' + variable.split('_')[1]])[:, cellidx]
                else:
                    y_model = np.array(model_data[variable])[:, cellidx]
                    y_base = np.array(base_data[variable])[:, cellidx]

                try:
                    interp_base = interp1d(t_base, y_base, bounds_error=False, fill_value="extrapolate")
                    y_base_interp = interp_base(t_model)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Interpolation échouée pour {filename}: {e}")
                    return None

                if metric == "MAE":
                    err = np.mean(np.abs(y_model - y_base_interp))
                elif metric == "MAE%":
                    err = 100 * np.mean(np.abs((y_model - y_base_interp) / (np.abs(y_base_interp) + 1e-12)))
                elif metric == "MAX%":
                    err = np.max(np.abs((y_model - y_base_interp) / (np.abs(y_base_interp) + 1e-12)))
                elif metric == "MAX":
                    err = np.max(np.abs(y_model - y_base_interp))
                elif metric == "AREA_ERR%":
                    area_model = trapz(y_model, t_model)
                    area_base = simps(y_base, t_base)
                    err = 100 * np.abs(area_model - area_base) / np.abs(area_base) if area_base != 0 else np.nan
                else:
                    raise NotImplementedError('metric not implemented')

                return (T_val, phi_val, err)

            if verbose:
                print(f"🔄 Traitement de {len(common_files)} fichiers pour {model_dir}...")

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_file)(fname) for fname in tqdm(common_files)
            )
            results = [r for r in results if r is not None]
            results = np.array(results)
            if len(results) == 0:
                raise RuntimeError("Aucun résultat valide obtenu — vérifier les fichiers.")

            T_vals = np.unique(results[:, 0])
            phi_vals = np.unique(results[:, 1])
            Z = np.full((len(phi_vals), len(T_vals)), np.nan)
            for T, phi, err in results:
                i = np.where(phi_vals == phi)[0][0]
                j = np.where(T_vals == T)[0][0]
                Z[i, j] = err

            return T_vals, phi_vals, Z

        # --- 📊 Récupération des données ---
        T_vals, phi_vals, Z1 = process_model(model_dir1)
        _, _, Z2 = process_model(model_dir2)

        # --- 🎨 Plot côte à côte ---
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        for ax, Z, title in zip(axes, [Z1, Z2], [title1, title2]):
            cmap = plt.get_cmap("coolwarm")
            norm = mpl.colors.LogNorm(vmin=np.nanmin(Z[Z > 0]), vmax=np.nanmax(Z)) if log_scale else None
            c = ax.imshow(
                Z,
                extent=[T_vals.min(), T_vals.max(), phi_vals.min(), phi_vals.max()],
                origin='lower',
                aspect='auto',
                cmap=cmap,
                norm=norm,
                interpolation='bilinear',
                rasterized=True
            )
            cs = ax.contour(T_vals, phi_vals, Z, colors='k', levels=levels, linewidths=0.3, alpha=0.4)
            ax.clabel(cs, inline=True, fmt="%.1f", fontsize=6)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.tick_params(which='both', direction='in')

        axes[0].set_ylabel(ylabel)

        # Position manuelle de la colorbar (x0, y0, width, height)
        cax = fig.add_axes([0.95, 0.13, 0.02, 0.91 - 0.15])  # <-- ajuster ici
        cbar = fig.colorbar(c, cax=cax)
        cbar.set_label(colorbarLabel)
        if log_scale:
            cbar.ax.yaxis.set_major_formatter(LogFormatter())
        else:
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        cbar.set_ticks([])

        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.15, top=0.91, wspace=0.11)

        if save_fig:
            fname = f"error_map_{model_dir1}_{model_dir2}_{variable}_{metric}"
            plt.savefig(fname)
            print(f"✅ Figure sauvegardée : {fname}")
        else:
            plt.show()

        return Z1, Z2
