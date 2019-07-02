from random import choice

from view import *
from model import Player, Particle, Killer
from network import Network

# Initial Values

number_of_killers = 4
number_of_particles = 3

windowWidth = 400
windowHeight = 400

n_cells = 1 + number_of_particles + number_of_killers

network = Network([2*n_cells, 100, n_cells-1])

# Automatic
window_dim = windowWidth, windowHeight
direction_list = [-1, 0, 1, 10]
particle_list = []
killer_list = []
player = Player(window_dim, network)

for i in range(0, number_of_particles):
    new_particle = Particle('particle_{}'.format(i), window_dim)
    particle_list.append(new_particle)
for i in range(0, number_of_killers):
    new_killer = Killer('killer_{}'.format(i), choice(direction_list), window_dim)
    killer_list.append(new_killer)

A = App(player, particle_list, killer_list, window_dim)
A.on_execute()

