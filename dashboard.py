from random import choice

from view import App
from model import Player, Particle, Killer
from network import Network


# -------- Set parameters -------- #
number_of_killers = 1
number_of_particles = 1
windowWidth = 500
windowHeight = windowWidth
# training rate eta
eta = 0.1
# Choose if you want the network to play (True) or yourself (False)
use_network = True
# Neural network properties
first_layer = 32
second_layer = 32
# third_layer = 9

# If the game is runnung too fast/slow on your machine you can lower/higher this value to tune the time.sleep() parameter.
time_sleep = 0.015

# Set a limit for the number of rounds the game will run.
round_limit = 1000

# Choose whether to render the game or not
render = True


# -------- Automatic parameters -------- #
n_cells = 1 + number_of_particles + number_of_killers

network = Network([2*n_cells, first_layer, second_layer, 2])

window_dim = windowWidth, windowHeight
direction_list = [-1, 0, 1, 10]
particle_list = []
killer_list = []

player = Player(window_dim, network, use_network)

for i in range(0, number_of_particles):
    new_particle = Particle(f'particle_{i}', window_dim)
    particle_list.append(new_particle)
for i in range(0, number_of_killers):
    new_killer = Killer(f'killer_{i}', choice(direction_list), window_dim)
    killer_list.append(new_killer)

A = App(player, particle_list, killer_list, window_dim, eta, time_sleep, round_limit, render)
A.on_execute()

with open('level_data.py', 'w') as file:
    file.write('data = [')

with open('level_data.py', 'a') as file:
    file.write('\n\t')
    file.write(',\n\t'.join(str(x) for x in player.level_data))
    file.write('\n]')

with open('level_data.py', 'a') as file:
    file.write('\nmax_level = ' + (str(max(player.level_data)) if player.level_data else '0'))

with open('level_data.py', 'a') as file:
    file.write('\nfinal_level = ' + str(player.level))

if player.use_network:
    with open("network_info_end.txt", "w") as file:
        file.write("BIASES:\n{}\n".format(player.network.biases))
        file.write("WEIGHTS:\n{}\n".format(player.network.weights))
