import pygame
import random as rn
import math
import numpy as np
import time

from pygame.locals import *

# Miscellaneous functions

def coll(player, obj, windowWidth, windowHeight):
    x, y = player.x, player.y
    dd = math.sqrt(((x - obj.x)**2) + ((y - obj.y)**2))
    if dd < 10:
        obj.x = rn.randint(0, windowWidth)
        obj.y = rn.randint(0, windowHeight)
        player.level_up()
        return 0
    return dd


def fight(player, obj, windowWidth, windowHeight):
    limit = max(player.size, obj.size)
    x, y = player.x, player.y
    dd = math.sqrt(((x - obj.x)**2) + ((y - obj.y)**2))
    if dd < 9:
        # give player high risk high reward chance to level up
        player.level_down(obj.level)
        # obj.level_up()
        return -100
    return dd


class App():
    '''Class that runs the game'''
    def __init__(self, player, particles, killers, window_dim, eta, time_sleep):
        self.windowWidth, self.windowHeight = window_dim
        self.player = player
        self.particles = particles
        self.killers = killers
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self.n_cells = 1 + len(particles) + len (killers)
        self.eta = eta
        self.time_sleep = time_sleep

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        pygame.display.set_caption('')
        self._running = True
        self.start_time = time.perf_counter()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    # def on_loop(self):
    #     pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        if isinstance(self.player.x, np.ndarray):
            self.player.x = float(self.player.x[0])
            self.player.y = float(self.player.y[0])

        pygame.draw.circle(
            self._display_surf,
            (self.player.R, self.player.G, self.player.B),
            (int(self.player.x), int(self.player.y)),
            self.player.size,
            self.player.fill,
        )

        diag = math.hypot(self.windowWidth, self.windowHeight)

        if self.player.use_network:
            # Input = relative positions of objects (normalized)
            coord_array = np.zeros((2*self.n_cells, 1), dtype=np.float32)
            coord_array[0] = self.player.x / self.windowWidth
            coord_array[1] = self.player.y / self.windowHeight

            scores = []
            rel_positions = []
            count = 1

            # --- FOOD (reward = +1) ---
            for obj in self.particles:
                distance = coll(self.player, obj, self.windowWidth, self.windowHeight)
                pygame.draw.circle(
                    self._display_surf, (obj.R, obj.G, obj.B), (int(obj.x), int(obj.y)), 4, 3
                )

                dx = (obj.x - self.player.x) / diag
                dy = (obj.y - self.player.y) / diag
                rel_positions.append((dx, dy))

                score = 1.0 if distance == 0 else 0.2  # simple positive reward
                scores.append(score)

                coord_array[2*count] = dx
                coord_array[2*count+1] = dy
                count += 1

            # --- KILLERS (reward = -1) ---
            for obj in self.killers:
                distance = fight(self.player, obj, self.windowWidth, self.windowHeight)
                pygame.draw.circle(
                    self._display_surf,
                    (obj.R, obj.G, obj.B),
                    (int(obj.x), int(obj.y)),
                    obj.size,
                    obj.fill,
                )

                dx = (obj.x - self.player.x) / diag
                dy = (obj.y - self.player.y) / diag
                rel_positions.append((dx, dy))

                score = -1.0 if distance == -100 else -0.2  # strong negative on collision
                scores.append(score)

                coord_array[2*count] = dx
                coord_array[2*count+1] = dy
                count += 1

            total = sum(abs(s) for s in scores) + 1e-6
            dx_target = sum(dx * s for (dx, dy), s in zip(rel_positions, scores)) / total
            dy_target = sum(dy * s for (dx, dy), s in zip(rel_positions, scores)) / total

            # Normalize to get a unit vector
            norm = max(abs(dx_target), abs(dy_target), 1e-6)
            y_target = np.array([[dx_target / norm], [dy_target / norm]], dtype=np.float32)

            # Train network
            activations = self.player.network.SGD((coord_array, y_target), 1, 1, self.eta)

            # Predicted movement
            dx_pred, dy_pred = activations.ravel()

            # Add small exploration noise (5%)
            if np.random.rand() < 0.05:
                dx_pred += np.random.uniform(-0.3, 0.3)
                dy_pred += np.random.uniform(-0.3, 0.3)

            # Normalize prediction
            # norm = max(abs(dx_pred), abs(dy_pred), 1e-6)
            norm = math.hypot(dx_target, dy_target) + 1e-6
            dx_pred /= norm
            dy_pred /= norm

            # Move player
            self.player.x = (self.player.x + self.player.speed * dx_pred) % self.windowWidth
            self.player.y = (self.player.y + self.player.speed * dy_pred) % self.windowHeight

        if self.player.level == 0:
            if self.player.use_network:
                with open("network_info_end", "w") as file:
                    file.write("BIASES:\n{}\n".format(self.player.network.biases))
                    file.write("WEIGHTS:\n{}\n".format(self.player.network.weights))
            self._running = False

        pygame.display.flip()


    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() is False:
            self._running = False

        while(self._running):
            time.sleep(self.time_sleep)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if (keys[K_RIGHT]):
                self.player.moveRight()
            if (keys[K_LEFT]):
                self.player.moveLeft()

            if (keys[K_UP]):
                self.player.moveUp()

            if (keys[K_DOWN]):
                self.player.moveDown()
            if (keys[K_ESCAPE]):
                self._running = False
            for obj in self.particles:
                    obj.moveRight()
                    obj.moveUp()

            for obj in self.killers:
                    obj.moveRight()
                    obj.moveUp()

            # self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
