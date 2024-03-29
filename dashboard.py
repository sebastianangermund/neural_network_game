import time

from random import choice

from view import *
from model import Player, Particle, Killer
from network import Network


# -------- Set parameters -------- #
number_of_killers = 6
number_of_particles = 6
windowWidth = 800
windowHeight = 800
# training rate eta
eta = 0.4
# Choose if you want the network to play (True) or yourself (False)
use_network = True
# Neural network properties
first_layer = 2
second_layer = 9
third_layer = 4

# If the game is runnung too fast/slow on your machine you can lower/higher this value to tune the time.sleep() parameter.
time_sleep = 0.01


# -------- Automatic parameters -------- #
n_cells = 1 + number_of_particles + number_of_killers
network = Network(
    [2*n_cells, first_layer, second_layer, third_layer, n_cells-1]
)
window_dim = windowWidth, windowHeight
direction_list = [-1, 0, 1, 10]
particle_list = []
killer_list = []
start_time = time.perf_counter()
player = Player(window_dim, network, use_network, start_time)

for i in range(0, number_of_particles):
    new_particle = Particle('particle_{}'.format(i), window_dim)
    particle_list.append(new_particle)
for i in range(0, number_of_killers):
    new_killer = Killer('killer_{}'.format(i), choice(direction_list),
                        window_dim)
    killer_list.append(new_killer)

A = App(player, particle_list, killer_list, window_dim, eta, time_sleep)
A.on_execute()

with open('time_data.py', 'w') as file:
    file.write('data = [')

with open('time_data.py', 'a') as file:
    file.write(',\n'.join(str(x) for x in player.time_data))
    file.write('\n]')
