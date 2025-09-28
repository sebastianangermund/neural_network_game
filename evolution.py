import numpy as np
import copy
from network import Network

class Evolution:
    def __init__(self, n_cells, population_size=20, mutation_rate=0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.networks = [Network([2*n_cells, 32, 16, 2]) for _ in range(population_size)]

    def evaluate(self, env_class, n_steps=500):
        """Run each network in the environment, return scores"""
        scores = []
        for net in self.networks:
            env = env_class(network=net, render=False)
            score = env.run(n_steps)
            scores.append(score)
        return scores

    def evolve(self, scores, elite_fraction=0.2):
        """Keep best networks, clone & mutate the rest"""
        n_elite = max(1, int(self.population_size * elite_fraction))
        elite_idx = np.argsort(scores)[::-1][:n_elite]  # best first
        new_networks = [copy.deepcopy(self.networks[i]) for i in elite_idx]

        while len(new_networks) < self.population_size:
            parent = copy.deepcopy(np.random.choice(new_networks))
            self.mutate(parent)
            new_networks.append(parent)

        self.networks = new_networks

    def mutate(self, net):
        """Add Gaussian noise to weights"""
        for w in net.weights:
            w += self.mutation_rate * np.random.randn(*w.shape)
        for b in net.biases:
            b += self.mutation_rate * np.random.randn(*b.shape)
