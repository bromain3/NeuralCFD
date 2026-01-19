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
    value_cedre = 'Temperature'
    value = 'T'
    # values names : Temperature, Pression, F1:P1:H2, RhoY_F1:P1:H2, VitesseX
    t_cedre, field_cedre = readCEDREDatabase(r'/stck/bromain/cedre1d/sod_choc/archive_CHARME_tube1D_inst.cgns', 'tube1D', 'champ', value_cedre)
    x, y, z = readCEDREDatabase(r'/stck/bromain/cedre1d/sod_choc/archive_CHARME_tube1D_geom.cgns', 'tube1D', 'geom')

    sim_data = loadData('../../simulation_results/sod_choc', 'sod_choc_rus_1st_order.json')

    func_field_cedre_for_interp = interp1d(t_cedre, field_cedre, axis=0, kind='linear', bounds_error=True)
    func_field_self_for_interp = interp1d(sim_data['t'], sim_data[value], axis=0, kind='linear', bounds_error=True)



    plot_times = np.linspace(sim_data['t'][0], sim_data['t'][-1], 400)
    # Interpolation sur les temps de la première simulation
    field_cedre_interp = func_field_cedre_for_interp(plot_times)
    field_self_interp = func_field_self_for_interp(plot_times)

    fig, ax = plt.subplots()
    graph_cedre = ax.plot(x, field_cedre_interp[0], label='CEDRE ')[0]
    graph = ax.plot(sim_data['x'], field_self_interp[0], label='piecwise constant Rusanov')[0]

    min_y = min([min(i) for i in field_self_interp] + [min(i) for i in field_cedre_interp])
    max_y = max([max(i) for i in field_self_interp] + [max(i) for i in field_cedre_interp])
    delta_y = max_y - min_y
    blank_proportion = 0.05
    plt.ylim(min_y - blank_proportion * delta_y, max_y + blank_proportion * delta_y)
    time = plot_times
    plt.title(value + f' | t = {time}')
    plt.legend()


    def update(frame):
        # creating a new graph or updating the graph
        graph.set_ydata(field_self_interp[frame])
        graph_cedre.set_ydata(field_cedre_interp[frame])
        time = plot_times[frame]
        plt.title(value + f' | t = {time:.6e}')


    anim = FuncAnimation(fig, update, frames=len(plot_times), interval=10)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=100,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    anim.save(os.path.join('../../simulation_results/sod_choc', f'first_order_rus_{value}.gif'), writer=writer)

    #plt.show()  # TODO: remove what is under
    plt.show(block=False)
    input("Animation terminée — appuie sur Entrée pour continuer...")
    plt.close(fig)
