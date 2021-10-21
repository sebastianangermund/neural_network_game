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
        if type(self.player.x) == np.ndarray:
            self.player.x = self.player.x[0]
            self.player.y = self.player.y[0]
        pygame.draw.circle(
            self._display_surf,
            (self.player.R, self.player.G, self.player.B),
            (self.player.x, self.player.y),
            self.player.size,
            self.player.fill,
        )
        if self.player.use_network:
            state_array = np.zeros((self.n_cells-1, 1))
            coord_array = np.zeros((2*self.n_cells, 1))
            coord_array[0] = self.player.x
            coord_array[1] = self.player.y
            count = 1
        for obj in self.particles:
            distance = coll(self.player, obj, self.windowWidth, self.windowHeight)
            pygame.draw.circle(
                self._display_surf, (obj.R, obj.G, obj.B), (obj.x, obj.y), 4, 3)
            if self.player.use_network:
                coord_array[2*count] = obj.x
                coord_array[2*count+1] = obj.y
                if distance == 0:
                    state_array[count-1] = -1
                else:
                    if not state_array[np.argmin(state_array)] == -1:
                        state_array[count-1] = 1 - distance/(self.windowWidth**2+self.windowHeight**2)**(1/2)
                count += 1

        for obj in self.killers:
            distance = fight(self.player, obj, self.windowWidth, self.windowHeight)
            pygame.draw.circle(
                self._display_surf,
                (obj.R, obj.G, obj.B),
                (obj.x, obj.y),
                obj.size,
                obj.fill,
            )
            if self.player.use_network:
                coord_array[2*count] = obj.x
                coord_array[2*count+1] = obj.y
                if not state_array[np.argmin(state_array)] == -1:
                    if distance == -100:
                        state_array[count-1] = -100
                    else:
                        state_array[count-1] = -2*distance/(self.windowWidth**2+self.windowHeight**2)**(1/2)
                count += 1

        if self.player.use_network:
            # OBS
            # If you give a random array as input, the player will only
            # chase a single particle. If you give the coordinates however,
            # it will sometimes switch to a different particle!
            # *GIVE RANDOM ARRAY AS INPUT*
            # state_array = np.random.uniform(low=0, high=self.windowWidth, size=state_array.shape)
            # *COMMENT OUT TO GIVE COORDINATES*
            activations = self.player.network.SGD((np.float32(coord_array), state_array), 1, 1, 0.3)
            choice_index = np.argmax(activations)
            x_direction = (coord_array[2*(choice_index+1)] - self.player.x)
            y_direction = (coord_array[2*(choice_index+1)+1] - self.player.y)
            factor = max(abs(x_direction), abs(y_direction))
            x_step = x_direction/factor
            y_step = y_direction/factor
            self.player.x = (self.player.x + 8*x_step) % self.windowWidth
            self.player.y = (self.player.y + 8*y_step) % self.windowHeight

        if self.player.level == 0:
            if self.player.use_network:
                file = open('network_info_end', 'w')
                file.write('BIASES: \n')
                file.write('{} \n'.format(self.player.network.biases))
                file.write('WEIGHTS: \n')
                file.write('{} \n'.format(self.player.network.weights))
                file.close()
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
