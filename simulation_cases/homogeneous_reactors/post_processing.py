import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from matplotlib import animation


import matplotlib
from cycler import cycler

N = 2
cmap = plt.get_cmap("coolwarm")
colors = [cmap(i / (N - 1)) for i in range(N)]


# 3. Créer un cycler propre
linestyles = ["-", "-"]


matplotlib.rcParams.update({
    # --- Latex ---
    'text.usetex': True,

    # --- Figure ---
    'figure.figsize': (3.5, 2.5),  # en pouces, bon format pour une colonne (~90 mm)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    #'savefig.bbox': 'tight',
    'savefig.format': 'png',

    # --- Police ---
    #'font.family': 'sans-serif',
    #'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'serif', # type cmr 10 : police latex
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # --- Axes ---
    'axes.linewidth': 0.8,
    #'axes.spines.top': False,
    #'axes.spines.right': False,
    'axes.grid': False,
    #'axes.labelpad': 4,
    'axes.formatter.use_mathtext': True,    # offset en mathtext (affiche ×10^{...})
    'axes.formatter.useoffset': True,       # garde l'offset, mais on forcera la forme scientifique
    'axes.formatter.limits': (-2, 2),
    'axes.axisbelow': True,  # pour que les grilles/patchs soient en dessous des lignes

    # --- Ticks ---
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    # --- Lignes ---
    'lines.linewidth': 0.8,
    'lines.markersize': 2.0,

    # --- Légendes ---
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    #'legend.handletextpad': 0.4,

    # --- Couleurs ---
    'image.cmap': 'coolwarm',
    'axes.prop_cycle': cycler(color=colors) + cycler(linestyle=linestyles)

})




def allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys


def completWith0s(input):
    if len(input) < 3:
        res = (3 - len(input)) * '0' + input
    else:
        res = input
    return(res)


def readCEDREDatabase(filename, caseName, type, value='Temperature'):
    #TODO: deduce type from file name (and casename also)
    with h5py.File(filename, 'r') as f:
        #print(allkeys(f))
        if type == 'histo':
            root = f'/Base/Locales/{caseName}'
            res = f[f'/Base/Locales/{caseName}']
            x = res[f'{root}/HORLOGE/DATES/Temps/ data'][()]
            y = res[f'{root}/ETATS/MOYENS/T/ data'][()]
            return x, y

        if type == 'champ':
            root = f'/Base/{caseName}#001'
            t = f[f'/Base/BaseIterativeData/TimeValues/ data'][()]
            T_s = []
            for n in range(len(t)):
                T_s.append(f[f'{root}/CHARME_VOLU#{completWith0s(str(n + 1))}/{value}/ data'][()])
            #print(allkeys(res))
            return t, np.array(T_s)

        if type == 'geom':
            root = f'/Base/{caseName}#001/.cedre#Geometry/CellCoordinates'
            x = f[f'{root}/CoordinateX/ data'][()]
            y = f[f'{root}/CoordinateY/ data'][()]
            z = f[f'{root}/CoordinateZ/ data'][()]
            return x, y, z


def loadData(dataPath, caseName):
    with open(os.path.join(dataPath, caseName), 'r') as f:
        simData = json.load(f)
    return(simData)

if __name__ == "__main__":
    value_cedre = 'Pression'
    value = 'P'
    # values names : Temperature, Pression, F1:P1:H2, RhoY_F1:P1:H2, VitesseX
    t_cedre, field_cedre = readCEDREDatabase(r'/stck/bromain/cedre1d/reactor_cst_v/archive_CHARME_CUBE_inst.cgns', 'CUBE', 'champ', value_cedre)

    sim_data = loadData('../../simulation_results/homogeneous_reactors', 'cst_rho.json')


    plt.plot(t_cedre, field_cedre, label='CEDRE cst_v')
    plt.plot(sim_data['t'], sim_data[value], label='model cst_v')

    plt.legend()
    plt.show()

