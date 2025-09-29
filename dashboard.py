import os

from random import choice

from view import App
from model import Player, Particle, Killer
from network import Network


class Config:
    # Number of particles and killers in the game
    number_of_killers = 2
    number_of_particles = 2
    windowWidth = 500
    windowHeight = windowWidth
    window_dim = windowWidth, windowHeight
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
    round_limit = 10000
    # Choose whether to render the game or not
    render = False


def get_player(conf):
    n_cells = 1 + conf.number_of_particles + conf.number_of_killers
    network = Network([2*n_cells, conf.first_layer, conf.second_layer, 2])
    player = Player(conf.window_dim, network, conf.use_network)
    return player


def get_npcs(conf):
    direction_list = [-1, 0, 1, 10]
    particle_list = []
    killer_list = []
    for i in range(0, conf.number_of_particles):
        new_particle = Particle(f'particle_{i}', conf.window_dim)
        particle_list.append(new_particle)
    for i in range(0, conf.number_of_killers):
        new_killer = Killer(f'killer_{i}', choice(direction_list), conf.window_dim)
        killer_list.append(new_killer)
    return particle_list, killer_list


def write_data(player, batch, instance, max_level, final_level):
    write_path = 'networks/' + 'batch_' + str(batch) + '/'
    stats_file = write_path + 'network' + str(instance) + '_stats.txt'
    net_file = write_path + 'network' + str(instance) + '.txt'

    os.makedirs(write_path, exist_ok=True)

    with open(stats_file, 'w') as file:
        file.write('data = [')

    with open(stats_file, 'a') as file:
        file.write('\n\t')
        file.write(',\n\t'.join(str(x) for x in player.level_data))
        file.write('\n]')

    with open(stats_file, 'a') as file:
        file.write('\nmax_level = ' + str(max_level))

    with open(stats_file, 'a') as file:
        file.write('\nfinal_level = ' + str(final_level))

    with open(net_file, 'w') as file:
        file.write('BIASES:\n{}\n'.format(player.network.biases))
        file.write('WEIGHTS:\n{}\n'.format(player.network.weights))

    return


def run_simulation_round(player, conf):
    particle_list, killer_list = get_npcs(conf)
    A = App(
        player,
        particle_list,
        killer_list,
        conf.window_dim,
        conf.eta,
        conf.time_sleep,
        conf.round_limit,
        conf.render
    )
    A.run()


def run_simulation_batch(n_rounds, batch):
    if conf.render is True:
        raise ValueError("If render is True, only run a single simulation.")
    survivors = []
    for i in range(n_rounds):
        player = get_player(conf)
        run_simulation_round(player, conf)
        max_level = max(player.level_data) if player.level_data else 0
        final_level = player.level
        # Only save strong networks
        if max_level >= 25 and final_level >= 1:
            survivors.append((player, max_level, final_level))
    return survivors


if __name__ == "__main__":

    conf = Config()

    if conf.render is True:
        player = get_player(conf)
        run_simulation_round(player, conf)
        exit()

    n_batches = 5
    n_rounds_per_batch = 20
    for batch in range(n_batches):
        print(f'Starting batch {batch+1} of {n_batches}')
        survivors = run_simulation_batch(n_rounds_per_batch, batch)
        print(f'Finished batch {batch+1} of {n_batches}\nFound {len(survivors)} survivors')
        # You can now use the survivors list for further processing

    if not survivors:
        print("No survivors found in any batch.")
        exit()

    ranked_networks = sorted(survivors, key=lambda x: (x[1] + x[2]), reverse=True)

    # Test the best network
    test_conf = Config()
    test_conf.render = True
    test_conf.round_limit = 5000
    player = ranked_networks[0][0]
    run_simulation_round(player, test_conf)
