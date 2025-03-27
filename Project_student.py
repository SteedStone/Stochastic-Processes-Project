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
    P = np.clip(P, 1e-10, 1)  # Évite log(0) et garde P dans une plage valide
    Q = np.clip(Q, 1e-10, 1)
    return np.sum(P * np.log(P / Q))

def bhattacharyya_distance(P, Q):
    bc = np.sum(np.sqrt(P * Q))
    return -np.log(np.clip(bc, 1e-10, 1))

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

def comparaison_of_resampling() : 
    

    def multinomial_resampling(weights):
        N = len(weights)
        indices = np.random.choice(N, N, p=weights)
        return indices

    def residual_resampling(weights):
        N = len(weights)
        num_copies = np.floor(N * weights).astype(int)
        residuals = weights * N - num_copies
        residuals /= residuals.sum()
        
        indices = np.hstack([
            np.repeat(i, num_copies[i]) for i in range(N)
        ])
        
        num_remaining = N - len(indices)
        if num_remaining > 0:
            resample_indices = np.random.choice(N, num_remaining, p=residuals)
            indices = np.hstack([indices, resample_indices])
        
        return indices

    def systematic_resampling(weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform()) / N
        cumulative_sum = np.cumsum(weights)
        indices = np.zeros(N, dtype=int)
        j = 0
        for i in range(N):
            while positions[i] > cumulative_sum[j]:
                j += 1
            indices[i] = j
        return indices

    # Génération des poids aléatoires
    N = 100  # Nombre de particules
    weights = np.random.rand(N)
    weights /= weights.sum()  # Normalisation

    # Application des méthodes de rééchantillonnage
    indices_multinomial = multinomial_resampling(weights)
    indices_residual = residual_resampling(weights)
    indices_systematic = systematic_resampling(weights)

    # Calcul de la variance de N_t^{(i)}
    def compute_variance(indices, N):
        counts = np.bincount(indices, minlength=N)
        return np.var(counts)

    var_multinomial = compute_variance(indices_multinomial, N)
    var_residual = compute_variance(indices_residual, N)
    var_systematic = compute_variance(indices_systematic, N)

    print(f"Variance Multinomial: {var_multinomial:.4f}")
    print(f"Variance Residual: {var_residual:.4f}")
    print(f"Variance Systematic: {var_systematic:.4f}")

    # Visualisation des résultats
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(indices_multinomial, bins=N, alpha=0.7)
    plt.title("Multinomial Resampling")

    plt.subplot(1, 3, 2)
    plt.hist(indices_residual, bins=N, alpha=0.7)
    plt.title("Residual Resampling")

    plt.subplot(1, 3, 3)
    plt.hist(indices_systematic, bins=N, alpha=0.7)
    plt.title("Systematic Resampling")

    plt.tight_layout()
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

affichage = "c"
# a) 
if affichage == "a" :
    plot_2d_heatmaps(box, box_length)
    plot_3d_scatter(box)

# c)
elif affichage == "c":
    plot_lorenz_with_sliders()
    hist1 = create_histogram(domain, box_length, sigma, rho, beta, state0, t)
    hist2 = create_histogram(domain, box_length, 5, 18, 8, state0, t)
    kl_div, bhattacharyya_dist = compare_distributions(hist1, hist2)
    print(f"KL Divergence: {kl_div}")
    print(f"Bhattacharyya Distance: {bhattacharyya_dist}")
# d)
elif affichage == "d":
    state1 = [10.0, 10.0, 10.0]
    hist1 = create_histogram(domain, box_length, sigma, rho, beta, state0, t)
    hist2 = create_histogram(domain, box_length, sigma, rho, beta, state1, t)

    kl_div, bhattacharyya_dist = compare_distributions(hist1, hist2)
    print(f"KL Divergence: {kl_div}")
    print(f"Bhattacharyya Distance: {bhattacharyya_dist}")
elif affichage == "e": 
    comparaison_of_resampling()


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
   
 