from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from adapenvs.farmworld.unit import Unit
from adapenvs.farmworld.utilities import layer_images, get_resource_path
import random

class Chicken(Unit):

    directions = np.array([[-1, 0], # UP (taken as y, x delta with 0,0 as top left)
                            [1, 0], # DOWN
                            [0, -1], # LEFT
                            [0, 1]]) # RIGHT

    rotation_directions = [180, 0, 270, 90]

    chicken = np.array(Image.open(get_resource_path("chicken.png")))

    all_chickens = []
    for r in rotation_directions:
        all_chickens.append(np.array(Image.fromarray(chicken).rotate(r)))

    def __init__(self, farmworld, max_health : int, loc:Union[None, Tuple[int, int]] = None):
        self._max_health = max_health
        self._health = self._max_health
        if loc != None:
            self._location = np.array([*loc])
            farmworld.available_locs.remove(loc)
        else:
            self._location : np.ndarray = farmworld.pop_available_xy_coord()

        self.orientation = random.randint(0, 3)
        self.current_adventure_length = 4

        # scale the encoding to be between 0 and 255 so we can use CNN policy structure
        self.encoding_scaling = 1/(np.array([Unit.ALL, self._max_health, 3])+1) * 255

        self.farmworld = farmworld
        self.upper_bounds = np.array([farmworld.sidelength_y - 1, farmworld.sidelength_x - 1])
        self.lower_bounds = np.array([0, 0])

        self.cool_down_count = farmworld.chicken_cooldown

    def _reset_chicken(self):
        self._location : np.ndarray = self.farmworld.pop_available_xy_coord()
        self.orientation = random.randint(0, 3)
        self._health = self._max_health
        self.cool_down_count = self.farmworld.chicken_cooldown

    def get_square_infront(self) -> Tuple[int, int]:
        '''
        Returns a tuple of the square "in front" of the agent's orientation direction,
        even if that square is out-of-bounds (i.e. may contain negative values)
        '''
        return tuple(self._location + Chicken.directions[self.orientation])

    @staticmethod
    def rotate(arr : np.ndarray, orientation):
        return np.array(Image.fromarray(arr).rotate(Chicken.rotation_directions[orientation]))

    def location(self) -> Tuple[int, int]:
        return tuple(self._location)

    def get_health(self) -> int:
        return self._health

    def modify_health(self, mod : int):
        self._health += mod

    def is_alive(self):
        return self._health > 0

    def is_type(self, unit_type : int):
        return Unit.CHICKEN == unit_type

    def on_exit_timestep(self):
        if self.is_alive():
            # only move the chicken with small probability
            if random.random() < self.farmworld.config["chicken_move_prob"]:
                if self.current_adventure_length >= 0:
                    self.current_adventure_length -= 1
                else:
                    self.current_adventure_length = 4
                    self.orientation = random.randint(0, 3)

                old_loc = self.location()
                self._location += Chicken.directions[self.orientation]
                if self.location() not in self.farmworld.available_locs:
                    self._location -= Chicken.directions[self.orientation]
                else:
                    self.farmworld.available_locs.add(old_loc)
                    self.farmworld.available_locs.remove(self.location())
                self._location = np.clip(self._location, a_min=self.lower_bounds, a_max=self.upper_bounds)
        else:
            if self.cool_down_count == self.farmworld.chicken_cooldown:
                self.farmworld.available_locs.add(self.location()) # when it first dies, add back it's location
            self.cool_down_count -= 1
            if self.cool_down_count < 0:
                self._reset_chicken()

    def get_encoding(self) -> np.ndarray:
        return np.array([0, 0, 0, 1, self._health/self._max_health, self.orientation/3])

        # return (np.array([Unit.CHICKEN, self._health, self.orientation])+1) * self.encoding_scaling

    def get_img_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        return Chicken.all_chickens[self.orientation]