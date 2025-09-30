# view.py
import pygame
import random as rn
import math
import numpy as np
import time

from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_ESCAPE


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
    x, y = player.x, player.y
    dd = math.hypot(x - obj.x, y - obj.y)
    collided = dd < 9
    if collided:
        player.level_down(obj.level)
        # return sentinel for collision (kept for compatibility with your scoring)
    return collided


class App():
    '''Class that runs the game'''
    def __init__(self, player, particles, killers, window_dim, eta, time_sleep, round_limit=1000, render=True, old_network=False):
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
        self.round_limit = round_limit
        self.round_count = 0
        self.render = render
        self._validate()
        self.old_network = old_network

    def _validate(self):
        if self.render is False and self.player.use_network is False:
            raise ValueError("If render is False, use_network must be True")

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        pygame.display.set_caption('Super Simple Cell Simulator')

    def _render_graphics(self):
        self._display_surf.fill((0, 0, 0))
        # Draw player
        pygame.draw.circle(
            self._display_surf,
            (self.player.R, self.player.G, self.player.B),
            (int(self.player.x), int(self.player.y)),
            self.player.size,
            self.player.fill,
        )
        # Draw particles
        for obj in self.particles:
            pygame.draw.circle(
                self._display_surf, (obj.R, obj.G, obj.B), (int(obj.x), int(obj.y)), 4, 3
            )
        # Draw killers
        for obj in self.killers:
            pygame.draw.circle(
                self._display_surf, (obj.R, obj.G, obj.B), (int(obj.x), int(obj.y)), obj.size, obj.fill,
            )
        pygame.display.flip()

    # >>> UPDATED: pure PyTorch online step
    def _run_network(self, coord_array, y_target):

        if self.old_network:
            # Train network
            activations = self.player.network.SGD((coord_array, y_target), 1, 1, self.eta)
            # Predicted movement
            dx_pred, dy_pred = activations.ravel()
        else:
            # Train online and get prediction
            pred = self.player.network.train_step(coord_array.ravel(), y_target.ravel())
            dx_pred, dy_pred = float(pred[0]), float(pred[1])

        # Add small exploration noise (5%)
        if np.random.rand() < 0.05:
            dx_pred += np.random.uniform(-0.3, 0.3)
            dy_pred += np.random.uniform(-0.3, 0.3)

        # Normalize prediction by ITS OWN norm
        pred_norm = math.hypot(dx_pred, dy_pred) + 1e-6
        dx_pred /= pred_norm
        dy_pred /= pred_norm

        # Move player
        self.player.x = (self.player.x + self.player.speed * dx_pred) % self.windowWidth
        self.player.y = (self.player.y + self.player.speed * dy_pred) % self.windowHeight

    def _update_npc_positions(self):
        # Input = relative positions of objects (normalized)
        coord_array = np.zeros((2*self.n_cells, 1), dtype=np.float32)
        coord_array[0] = self.player.x / self.windowWidth
        coord_array[1] = self.player.y / self.windowHeight

        scores = []
        rel_positions = []
        count = 1

        diag = math.hypot(self.windowWidth, self.windowHeight)
        for obj in self.particles:
            distance = coll(self.player, obj, self.windowWidth, self.windowHeight)

            dx = (obj.x - self.player.x) / diag
            dy = (obj.y - self.player.y) / diag
            rel_positions.append((dx, dy))

            score = 1.0 if distance == 0 else 0.2  # positive toward particles
            scores.append(score)

            coord_array[2*count] = dx
            coord_array[2*count+1] = dy
            count += 1

            obj.moveRight()
            obj.moveUp()

        for obj in self.killers:
            collided = fight(self.player, obj, self.windowWidth, self.windowHeight)

            dx = (obj.x - self.player.x) / diag
            dy = (obj.y - self.player.y) / diag
            rel_positions.append((dx, dy))

            score = -1.0 if collided else -0.2  # negative away from killers
            scores.append(score)

            coord_array[2*count] = dx
            coord_array[2*count+1] = dy
            count += 1

            obj.moveRight()
            obj.moveUp()

        # >>> NEW: compute target ONCE after loops
        total = sum(abs(s) for s in scores) + 1e-6
        dx_target = sum(dx * s for (dx, _dy), s in zip(rel_positions, scores)) / total
        dy_target = sum(dy * s for (_dx, dy), s in zip(rel_positions, scores)) / total

        # L2 normalize for a unit direction
        norm = math.hypot(dx_target, dy_target) + 1e-6
        y_target = np.array([[dx_target / norm], [dy_target / norm]], dtype=np.float32)

        return coord_array, y_target

    def on_render(self):
        if isinstance(self.player.x, np.ndarray):
            self.player.x = float(self.player.x[0])
            self.player.y = float(self.player.y[0])

        coord_array, y_target = self._update_npc_positions()

        if self.player.use_network:
            self._run_network(coord_array, y_target)

        if self.player.level == 0:
            self._running = False

        if self.render:
            self._render_graphics()

    def on_cleanup(self):
        pygame.quit()

    def run(self):
        if self.render:
            if self.on_init() is False:
                self._running = False

        while(self._running):
            if self.render:
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

            self.round_count += 1
            if self.round_count >= self.round_limit:
                self._running = False

            self.on_render()

        if self.render:
            self.on_cleanup()
