# -*- coding: utf-8 -*-
"""
LINMA1731 Stochastic Processes

Code for the project

@author: Philémon Beghin and Amir Mehrnoosh
"""

"""
LORENZ SYSTEM
"""

# from https://en.wikipedia.org/wiki/Lorenz_system

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi
import random
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.stats import entropy, wasserstein_distance



sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# Lorenz model

def Lorenz(state,t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0] # initial condition
t = np.arange(0.0, 100.0, 0.02) # time vector

states = odeint(Lorenz, state0, t) # vector containing the (x,y,z) positions for each time step


def is_in_box(x, y, z, box):
    """
    Check if the point (x, y, z) is inside the box.
    """
    x0, x1 = box[0][0] , box[0][1]
    y0, y1 = box[1][0] , box[1][1]
    z0, z1 = box[2][0] , box[2][1]
   
    return (x0 <= x <= x1) and (y0 <= y <= y1) and (z0 <= z <= z1)
    
def create_box(domain , box_length):
    """
    Create boxes in the domain with the given box length.
    """
    boxes = {}
    x0 = domain[0][0] 
    x1 = domain[0][0]
    y0 = domain[1][0]
    y1 = domain[1][0] 
    z0 = domain[2][0] 
    z1 = domain[2][0] 
    for i in range((domain[0][1] - domain[0][0])//box_length):
        for j in range((domain[1][1] - domain[1][0])//box_length):
            for k in range((domain[2][1] - domain[2][0])//box_length):
                x0 = domain[0][0] + i * box_length
                x1 = x0 + box_length
                y0 = domain[1][0] + j * box_length
                y1 = y0 + box_length
                z0 = domain[2][0] + k * box_length
                z1 = z0 + box_length

                # Initialiser la boîte avec un compteur à 0
                boxes[((x0, x1), (y0, y1), (z0, z1))] = 0
    return boxes


def plot_3d_scatter(boxes):
    """
    Affiche un nuage de points 3D où la couleur indique la densité.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = []
    y_vals = []
    z_vals = []
    density = []
    boxes = {k: v for k, v in boxes.items() if v > 0}

    for ((x0, x1), (y0, y1), (z0, z1)), count in boxes.items():
        x_vals.append((x0 + x1) / 2)  # Position au centre de la boîte
        y_vals.append((y0 + y1) / 2)
        z_vals.append((z0 + z1) / 2)
        density.append(count)

    sc = ax.scatter(x_vals, y_vals, z_vals, c=density, cmap='plasma', s=30)
    plt.colorbar(sc, label="Nombre de points")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Distribution des points en 3D")
    
    plt.show()

def plot_2d_heatmaps(boxes, box_length):
    """
    Affiche des heatmaps de densité projetées sur les plans XY, XZ et YZ.
    """
    # Filtrer les boîtes où count > 0
    non_empty_boxes = {k: v for k, v in boxes.items() if v > 0}

    xy_grid = {}
    xz_grid = {}
    yz_grid = {}

    for ((x0, x1), (y0, y1), (z0, z1)), count in non_empty_boxes.items():
        xy_grid[(x0, y0)] = xy_grid.get((x0, y0), 0) + count
        xz_grid[(x0, z0)] = xz_grid.get((x0, z0), 0) + count
        yz_grid[(y0, z0)] = yz_grid.get((y0, z0), 0) + count

    def plot_heatmap(grid, xlabel, ylabel, title):
        if not grid:
            print(f"Aucun point pour {title}")
            return

        x_vals = [k[0] for k in grid.keys()]
        y_vals = [k[1] for k in grid.keys()]
        density = [grid[k] for k in grid.keys()]

        # Définir la grille de l'image
        x_bins = np.arange(min(x_vals), max(x_vals) + box_length, box_length)
        y_bins = np.arange(min(y_vals), max(y_vals) + box_length, box_length)

        # Créer la matrice de densité
        density_matrix = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        for (x, y), d in grid.items():
            i = np.searchsorted(x_bins, x) - 1
            j = np.searchsorted(y_bins, y) - 1
            if 0 <= i < density_matrix.shape[1] and 0 <= j < density_matrix.shape[0]:
                density_matrix[j, i] = d  # Matplotlib indexe en (ligne, colonne)

        plt.figure(figsize=(6, 5))
        plt.imshow(density_matrix, origin='lower', cmap='hot', extent=[min(x_bins), max(x_bins), min(y_bins), max(y_bins)], aspect='auto')
        plt.colorbar(label="Nombre de points")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    # Afficher les heatmaps pour les trois projections
    plot_heatmap(xy_grid, "X", "Y", "Projection XY (heatmap)")
    plot_heatmap(xz_grid, "X", "Z", "Projection XZ (heatmap)")
    plot_heatmap(yz_grid, "Y", "Z", "Projection YZ (heatmap)")

def plot_2d_projections(boxes):
    """
    Affiche des histogrammes de densité projetés sur les plans XY, XZ et YZ.
    Ignore les boîtes avec 0 points.
    """
    # Filtrer les boîtes où count > 0
    non_empty_boxes = {k: v for k, v in boxes.items() if v > 0}

    xy_counts = {}
    xz_counts = {}
    yz_counts = {}

    for ((x0, x1), (y0, y1), (z0, z1)), count in non_empty_boxes.items():
        xy_counts[(x0, y0)] = xy_counts.get((x0, y0), 0) + count
        xz_counts[(x0, z0)] = xz_counts.get((x0, z0), 0) + count
        yz_counts[(y0, z0)] = yz_counts.get((y0, z0), 0) + count

    def plot_projection(counts, xlabel, ylabel, title):
        if not counts:
            print(f"Aucun point pour la projection {title}")
            return

        x_vals = [k[0] for k in counts.keys()]
        y_vals = [k[1] for k in counts.keys()]
        density = list(counts.values())

        plt.figure(figsize=(6, 5))
        plt.scatter(x_vals, y_vals, c=density, cmap='viridis', s=50)
        plt.colorbar(label="Nombre de points")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    # Afficher les trois projections
    plot_projection(xy_counts, "X", "Y", "Projection sur le plan XY")
    plot_projection(xz_counts, "X", "Z", "Projection sur le plan XZ")
    plot_projection(yz_counts, "Y", "Z", "Projection sur le plan YZ")

def dict_to_histogram(boxes, domain, box_length):
    # Taille de la grille en 3D
    x_bins = (domain[0][1] - domain[0][0]) // box_length
    y_bins = (domain[1][1] - domain[1][0]) // box_length
    z_bins = (domain[2][1] - domain[2][0]) // box_length

    # Création d'un histogramme vide
    hist = np.zeros((x_bins, y_bins, z_bins))

    # Remplir l'histogramme avec les valeurs du dictionnaire
    for ((x0, x1), (y0, y1), (z0, z1)), count in boxes.items():
        i = int((x0 - domain[0][0]) / box_length)
        j = int((y0 - domain[1][0]) / box_length)
        k = int((z0 - domain[2][0]) / box_length)
        hist[i, j, k] = count  # Mettre le nombre de points

    return hist

def normalize_pdf(hist):
    return hist / np.sum(hist)

def kl_divergence(P, Q):
    P = P + 1e-10  # Évite log(0)
    Q = Q + 1e-10
    return 0.5 * (np.sum(P * np.log(P / Q)) + np.sum(Q * np.log(Q / P)))

def bhattacharyya_distance(P, Q):
    return -np.log(np.sum(np.sqrt(P * Q)))

def compare_distributions(hist1, hist2):
    """
    Compare two distributions using KL divergence and Bhattacharyya distance.
    """
    hist1_normalized = normalize_pdf(hist1)
    hist2_normalized = normalize_pdf(hist2)

    kl_div = kl_divergence(hist1_normalized, hist2_normalized)
    bhattacharyya_dist = bhattacharyya_distance(hist1_normalized, hist2_normalized)

    return kl_div, bhattacharyya_dist

def create_histogram(domain, box_length , sigma , rho , beta , state0 , t):
    """
    Create a histogram of the Lorenz system in the given domain with the specified box length.
    """
    # Create boxes
    boxes = create_box(domain, box_length)

    # Compute the Lorenz system
    true_states = odeint(lorenz, state0, t, args=(sigma, rho, beta))

    # Compute the number of points in each box
    for i in range(len(states)):
        x = true_states[i][0]
        y = true_states[i][1]
        z = true_states[i][2]
        for b in boxes.keys():
            if is_in_box(x, y, z, b):
                boxes[b] += 1

    # Convert dictionary to histogram
    hist = dict_to_histogram(boxes, domain, box_length)

    return hist


def lorenz(state, t, sigma, rho, beta):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
# Here we mdoifiy a bit the function to observe the effect of the parameters on the system
# We will use the sliders to modify the parameters of the Lorenz system
def plot_lorenz_with_sliders():
    """
    Plot the Lorenz attractor with sliders to modify the parameters.
    """
    

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    state0 = [1.0, 1.0, 1.0]  
    t = np.linspace(0, 50, 5000) 


    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(221, projection='3d')  # Attracteur 3D
    ax_xy = fig.add_subplot(222)  # Projection XY
    ax_xz = fig.add_subplot(223)  # Projection XZ
    ax_yz = fig.add_subplot(224)  # Projection YZ

    # Fonction pour mettre à jour les graphes
    def update_plot(val):
        sigma = slider_sigma.val
        rho = slider_rho.val
        beta = slider_beta.val

        # Résolution des EDOs avec les nouveaux paramètres
        states = odeint(lorenz, state0, t, args=(sigma, rho, beta))
        x, y, z = states[:, 0], states[:, 1], states[:, 2]

        # Mise à jour des tracés
        ax3d.clear()
        ax3d.plot(x, y, z, color='blue', linewidth=0.5)
        ax3d.set_title("Attracteur de Lorenz (3D)")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        ax_xy.clear()
        ax_xy.plot(x, y, color='red', linewidth=0.5)
        ax_xy.set_title("Projection XY")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")

        ax_xz.clear()
        ax_xz.plot(x, z, color='green', linewidth=0.5)
        ax_xz.set_title("Projection XZ")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")

        ax_yz.clear()
        ax_yz.plot(y, z, color='purple', linewidth=0.5)
        ax_yz.set_title("Projection YZ")
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")

        fig.canvas.draw_idle()

    # Ajout des sliders pour modifier les paramètres
    ax_slider_sigma = plt.axes([0.15, 0.01, 0.3, 0.02])
    ax_slider_rho = plt.axes([0.55, 0.01, 0.3, 0.02])
    ax_slider_beta = plt.axes([0.15, 0.04, 0.3, 0.02])

    slider_sigma = Slider(ax_slider_sigma, 'Sigma', 0, 20, valinit=sigma)
    slider_rho = Slider(ax_slider_rho, 'Rho', 0, 40, valinit=rho)
    slider_beta = Slider(ax_slider_beta, 'Beta', 0, 3, valinit=beta)

    # Mise à jour des tracés lorsque les sliders changent
    slider_sigma.on_changed(update_plot)
    slider_rho.on_changed(update_plot)
    slider_beta.on_changed(update_plot)

    # Première mise à jour
    update_plot(None)

    plt.show()
        
# Main part of the code 
# We will compute and increment the number of times a point is in a box of interval 5.
# Pour stocker les boxs on va faire un grand dictionnaire et les box seront stockés sous forme de tuple ([x0 , x1], [y0 , y1], [z0 , z1])
domain = ([-20, 20], [-30, 30], [0, 50])
box_length = 5
box ={}
box = create_box(domain, box_length)
for i in range(len(states)):
    x = states[i][0]
    y = states[i][1]
    z = states[i][2]
    for b in box.keys():
        if is_in_box(x, y, z, b):
            box[b] += 1

#We verify that we have 5000 points 
sum = 0
for key in box.keys():
    sum += box[key]

# a) 
# plot_2d_heatmaps(box, box_length)
# plot_3d_scatter(box)

# c)
plot_lorenz_with_sliders()
hist1 = create_histogram(domain, box_length, sigma, rho, beta, state0, t)
hist2 = create_histogram(domain, box_length, 5, 18, 8, state0, t)
kl_div, bhattacharyya_dist = compare_distributions(hist1, hist2)
print(f"KL Divergence: {kl_div}")
print(f"Bhattacharyya Distance: {bhattacharyya_dist}")
# d)
state1 = [10.0, 10.0, 10.0]
hist1 = create_histogram(domain, box_length, sigma, rho, beta, state0, t)
hist1 = create_histogram(domain, box_length, sigma, rho, beta, state1, t)
kl_div, bhattacharyya_dist = compare_distributions(hist1, hist2)
print(f"KL Divergence: {kl_div}")
print(f"Bhattacharyya Distance: {bhattacharyya_dist}")


# fig = plt.figure()
# plt.rcParams['font.family'] = 'serif'
# ax = fig.add_subplot(projection="3d")
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['True system'])
# plt.draw()
# plt.show()


# """
# PLOTLY : TRUE SYSTEM
# """

# # Uncomment this section once you've installed the "Plotly" package

# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "browser"


# fig = go.Figure(data=[go.Scatter3d(x=states[:, 0],y=states[:, 1],z=states[:, 2],
#                                    mode='markers',
#                                    marker=dict(
#                                        size=2,
#                                        opacity=0.8
#     )                        
#                                    )])
# fig.update_layout(
#     title='True system')
# fig.update_scenes(aspectmode='data')
# fig.show()
   
 