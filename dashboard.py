from random import choice

from view import *
from model import Player, Particle, Killer
from network import Network

'''
Run the game with this script by running:
 $ python dashboard.py

'''

# -------- Set parameters -------- #
number_of_killers = 6
number_of_particles = 5
windowWidth = 450
windowHeight = 450
# training rate eta
eta = 0.4
# Choose if you want the network to play (True) or yourself (False)
use_network = True
# Neural network properties
first_layer = 20
second_layer = 99
third_layer = 30


# -------- Automatic parameters -------- #
n_cells = 1 + number_of_particles + number_of_killers
network = Network([2*n_cells, first_layer, second_layer, third_layer, n_cells-1])
window_dim = windowWidth, windowHeight
direction_list = [-1, 0, 1, 10]
particle_list = []
killer_list = []
player = Player(window_dim, network, use_network)

for i in range(0, number_of_particles):
    new_particle = Particle('particle_{}'.format(i), window_dim)
    particle_list.append(new_particle)
for i in range(0, number_of_killers):
    new_killer = Killer('killer_{}'.format(i), choice(direction_list), window_dim)
    killer_list.append(new_killer)

A = App(player, particle_list, killer_list, window_dim, eta)
A.on_execute()
