# dashboard.py
import os

from random import choice

from view import App
from model import Player, Particle, Killer
from network import Network         # <-- old
from torch_network import TorchSteeringNet, TorchNetConfig  # <-- NEW


class Config:
    # Number of particles and killers in the game
    number_of_killers = 1
    number_of_particles = 1
    windowWidth = 800
    windowHeight = windowWidth
    window_dim = windowWidth, windowHeight
    # training rate eta (kept for compat, unused by torch net)
    eta = 0.1
    # Choose if you want the network to play (True) or yourself (False)
    use_network = True
    # Neural network properties
    first_layer = 32
    second_layer = 32
    # PyTorch LR
    lr = 1e-3
    # If the game is runnung too fast/slow on your machine you can lower/higher this value to tune the time.sleep() parameter.
    time_sleep = 0.015
    # Set a limit for the number of rounds the game will run.
    round_limit = 12000
    # Choose whether to render the game or not
    render = False
    old_network = False  # If True, use old Network class; if False, use TorchSteeringNet


def get_player(conf):
    n_cells = 1 + conf.number_of_particles + conf.number_of_killers
    input_dim = 2 * n_cells  # matches coord_array length

    if conf.old_network:
        network = Network([2*n_cells, conf.first_layer, conf.second_layer, 2])  # old
    else:
        # NEW: PyTorch network
        net_cfg = TorchNetConfig(
            input_dim=input_dim,
            hidden1=conf.first_layer,
            hidden2=conf.second_layer,
            lr=conf.lr,
            device=None,  # auto
        )
        network = TorchSteeringNet(net_cfg)  # NEW

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


def write_data(player, instance, max_level, final_level):
    write_path = 'networks/' + '/'
    stats_file = write_path + 'network' + str(instance) + '_stats.txt'
    net_file = write_path + 'network' + str(instance) + '.pt'   # <-- save as .pt

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

    # Save PyTorch weights
    try:
        player.network.save(net_file)
    except Exception as e:
        print(f"Warning: failed to save network weights: {e}")

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
        conf.render,
        conf.old_network,
    )
    A.run()


def run_simulation_batch(players, level_cutoff=25):
    if conf.render is True:
        raise ValueError("If render is True, only run a single simulation.")
    survivors = []
    print(f'Running batch of {len(players)} players')
    for player in players:
        run_simulation_round(player, conf)
        max_level = max(player.level_data) if player.level_data else 0
        final_level = player.level
        # Only save strong networks
        if max_level >= level_cutoff and final_level >= 1:
            survivors.append((player, max_level, final_level))
    return survivors


def run_evolution(n_players, n_batches, level_cutoff):
    players = [get_player(conf) for _ in range(n_players)]
    for batch in range(n_batches):
        print(f'Starting batch {batch+1} of {n_batches}')
        survivors = run_simulation_batch(players, level_cutoff)
        players = [s[0] for s in survivors]  # keep only the Player instances
        if not survivors:
            print(f'No survivors found in batch {batch+1}, stopping.')
            exit()
        print(f'Finished batch {batch+1} of {n_batches}\nFound {len(survivors)} survivors')
        # You can now use the survivors list for further processing
    ranked_networks = sorted(survivors, key=lambda x: (x[1] + x[2]), reverse=True)
    return ranked_networks


if __name__ == "__main__":

    conf = Config()

    if conf.render is True:
        player = get_player(conf)
        run_simulation_round(player, conf)
        exit()

    n_players = 10
    n_batches = 3
    level_cutoff = 25
    ranked_networks = run_evolution(n_players, n_batches, level_cutoff)

    # Test the best network
    test_conf = Config()
    test_conf.render = True
    test_conf.round_limit = 5000
    player = ranked_networks[0][0]
    run_simulation_round(player, test_conf)

    # save the best network
    write_data(player, instance='best', max_level=ranked_networks[0][1], final_level=ranked_networks[0][2])
