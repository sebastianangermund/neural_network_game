import random as rn
import weakref
import math
import time
from random import randrange

from network import Network


class Particle:
    _instances = set()

    def __init__(self, name, window_dim):
        self.windowWidth, self.windowHeight = window_dim
        self.x = rn.randint(20, self.windowWidth-20)
        self.y = rn.randint(0, self.windowHeight)
        self.name = name
        self._instances.add(weakref.ref(self))
        self.R = randrange(0, 255)
        self.G = randrange(0, 255)
        self.B = randrange(0, 255)

    def moveRight(self):
        self.x = (self.x + rn.randint(-4, 4)) % self.windowWidth

    def moveUp(self):
        self.y = (self.y + rn.randint(-4, 4)) % self.windowHeight

    def delete(self, object):
        del self

    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead


class Killer:
    _instances = set()

    def __init__(self, name, direction, window_dim):
        self.name = name
        self.windowWidth, self.windowHeight = window_dim
        self.x = rn.randint(self.windowWidth-(self.windowWidth*0.2),
                            self.windowWidth)
        self.y = rn.randint(0, self.windowHeight)
        self.level = 1
        self.size = 3
        self.fill = 1
        self.R = 255
        self.G = 0
        self.B = 0
        self.direction = direction
        self._instances.add(weakref.ref(self))

    def moveRight(self):
        if self.direction == -1:
            self.x = (self.x + rn.randint(-9, -5)) % self.windowWidth
        elif self.direction == 0:
            self.x = (self.x + rn.randrange(5, 9)) % self.windowWidth
        else:
            self.x = (self.x + rn.randrange(-4, 4)) % self.windowWidth

    def moveUp(self):
        if self.direction == 1:
            self.y = (self.y + rn.randint(-9, -5)) % self.windowHeight
        elif self.direction == 10:
            self.y = (self.y + rn.randint(5, 10)) % self.windowHeight
        else:
            self.y = (self.y + rn.randrange(-4, 4)) % self.windowHeight

    def delete(self, object):
        del self

    def level_up(self):
        if self.G == 255:
            return
        self.level += 3
        if self.size < 10:
            self.size += 1
        elif self.fill < 10:
            self.fill += 1
        elif self.R < 15:
            self.R -= 10
            self.G += 10
        else:
            self. R = 0
            self.G = 255

    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead  # Not in use


class Player:
    _instances = set()

    def __init__(self, window_dim, network, use_network, start_time):
        self.windowWidth, self.windowHeight = window_dim
        self.network = network
        self.level = 20
        self.size = 3
        self.fill = 1
        self.x = 10
        self.y = int(math.floor(self.windowHeight * 0.5))
        self.R = 255
        self.G = 255
        self.B = 0
        self.speed = 8
        self.use_network = use_network
        self.start_time= start_time
        self.time_data = []

    def moveRight(self):
        self.x = (self.x + self.speed) % self.windowWidth

    def moveLeft(self):
        self.x = (self.x - self.speed) % self.windowWidth

    def moveUp(self):
        self.y = (self.y - self.speed) % self.windowHeight

    def moveDown(self):
        self.y = (self.y + self.speed) % self.windowHeight

    def level_up(self):
        if self.B == 255:
            return
        self.level += 1
        time_ = time.perf_counter() - self.start_time
        self.time_data.append((self.level, time_))
        if self.level % 10 == 0:
            if self.size < 10:
                self.size += 1
            elif self.fill < 10:
                self.fill += 1
            elif self.R > 10:
                self.R -= 2
                self.G -= 2
            else:
                self.R = 0
                self.B = 255

    def level_down(self, num):
        if self.level == 0:
            return
        if self.level - num >= 0:
            self.level -= num
        else:
            self.level = 0
        time_ = time.perf_counter() - self.start_time
        self.time_data.append((self.level, time_))
        if self.R < 250:
            self.R += 2
            self.G += 2
        elif self.fill > 1:
            self.fill -= 1
        elif self.size > 3:
            self.size -= 1
        else:
            self.B = 0
