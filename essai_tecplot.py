import tecplot as tp
import tecplot.constant as const
import matplotlib.pyplot as plt

import numpy as np
import csv

# ------------------------------
# Connexion à Tecplot
# ------------------------------
# Avec interface GUI :
tp.session.connect()
# Sans interface graphique :
# tp.session.connect(tp.session.ConnectionMode.Standalone)

# ------------------------------
# Charger le fichier Tecplot
# ------------------------------
file_path = r'D:\stck\cedre1d\sod_choc\_TECPLOT_\archive_CHARME.volu.ins.plt'
tp.data.load_tecplot(file_path)


# Récupérer le dataset actif
ds = tp.active_frame().dataset

# Afficher tous les time steps disponibles
print(ds.solution_times)  # liste de float : [0.0, 0.01, 0.02, ...]
times = list(ds.solution_times)
# Choisir le pas de temps le plus proche de t_target
t_target = 0.001
#closest_time = min(ds.solution_times, key=lambda t: abs(t - t_target))

# Sélectionner ce pas de temps dans le dataset
#ds.active_solution_time = closest_time

# Lister toutes les zones avec leur time
# for z in ds.zones():
#    print(z.name, z.solution_time)

# Sélectionner la zone la plus proche du temps demandé
closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - t_target))
print('zone selection')
zone = ds.zone(closest_idx)
#zone = min(ds.zones(), key=lambda z: abs(z.solution_time - t_target))
print(f"Zone choisie : {zone.name}, time = {zone.solution_time}")


# ------------------------------
# Paramètres de la ligne à extraire
# ------------------------------
alpha = 0.5        # y = alpha
xmin, xmax = 0.0, 1.0
npts = 200

# Nom de la variable que l'on veut extraire
var_name = 'P'  # <-- à changer selon ton fichier

# ------------------------------
# Extraire la ligne
# ------------------------------
# On va stocker les résultats
results = []

t = zone.solution_time  # pas de temps de la zone

# Récupérer les coordonnées des centres de cellules
# (pour les zones ordonnées)
values = zone.values(var_name).as_numpy_array()
x = zone.values('X').as_numpy_array()
y = zone.values('Y').as_numpy_array()
# Si tu veux filtrer les cellules autour d'une ligne y=alpha
#mask = np.abs(y - alpha) < 1e-6  # tolérance
#x_line = x[mask]
#values_line = values[mask]

_, unique_idx = np.unique(x, return_index=True)
x = x[unique_idx]
sorted_idx = np.argsort(x)
x = (x[:-1] + x[1:]) / 2


results.append({
    'time': t,
    'x': x,
    'values': values
})

# Exemple : afficher le profil de la première zone
print(f"Time = {results[0]['time']}")
for xi, vi in zip(results[0]['x'], results[0]['values']):
    print(f"x = {xi:.4f}, {var_name} = {vi:.6f}")

plt.plot(results[0]['x'], results[0]['values'])
plt.show()
# Optionnel : sauvegarder tout dans un CSV
with open(r'D:\stck\cedre1d\sod_choc\_TECPLOT_\extract.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'x', var_name])
    for r in results:
        for xi, vi in zip(r['x'], r['values']):
            writer.writerow([r['time'], xi, vi])