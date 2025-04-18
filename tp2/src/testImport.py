import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Assure-toi d'importer Axes3D ici

# Créer une figure et un axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Données de test pour un graphique 3D
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
z = [1, 8, 27, 64]

# Tracer
ax.plot(x, y, z)

# Afficher
plt.show()
