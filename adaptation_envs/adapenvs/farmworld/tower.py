from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from adapenvs.farmworld.unit import Unit
from adapenvs.farmworld.utilities import get_resource_path

class Tower(Unit):

    tower_base_img = np.array(Image.open(get_resource_path("castlebase.png")))
    tower_d1 = np.array(Image.open(get_resource_path("castled1.png")))
    tower_d2 = np.array(Image.open(get_resource_path("castled2.png")))
    tower_d3 = np.array(Image.open(get_resource_path("castled3.png")))
    tower_health_indicators = [tower_base_img, tower_d1, tower_d2, tower_d3]

    food_base_img = np.array(Image.open(get_resource_path("foodbase.png")))
    food_d1 = np.array(Image.open(get_resource_path("food1.png")))
    food_d2 = np.array(Image.open(get_resource_path("food2.png")))
    food_d3 = np.array(Image.open(get_resource_path("food3.png")))
    food_health_indicators = [food_base_img, food_d1, food_d2, food_d3]

    def __init__(self, farmworld, max_health: int, loc:Union[None, Tuple[int, int]] = None):
        self._max_health = max_health
        self.farmworld = farmworld

        self._is_defended = True
        self._health = self._max_health
        
        if loc != None:
            self._location = np.array([*loc])
            farmworld.available_locs.remove(loc)
        else:
            self._location = farmworld.pop_available_xy_coord()

        # scale the encoding to be between 0 and 255 so we can use CNN policy structure
        # in this case the last one is defended or not, so max val is 1
        self.encoding_scaling = 1/(np.array([Unit.ALL, self._max_health, 1])+1) * 255
        
        self.cool_down_count = self.farmworld.tower_cooldown

    def _reset_tower(self):
        self._health = self._max_health
        self._is_defended = True
        
        # old_loc = self.location()
        # self.farmworld.available_locs.add(old_loc)

        self._location = self.farmworld.pop_available_xy_coord()

        self.cool_down_count = self.farmworld.tower_cooldown

    def location(self) -> Tuple[int, int]:
        return tuple(self._location)

    def modify_health(self, mod : int):
        self._health += mod
        if self._health <= 0 and self._is_defended:
            self._is_defended = False
            self._health = self._max_health
        # elif self._health <= 0:
            # self._reset_tower()
            # self.farmworld.add

    def get_health(self) -> int:
        return self._health

    def is_defended(self):
        return self._is_defended
    
    def is_alive(self):
        return self._is_defended or self._health > 0
    
    def is_type(self, unit_type : int):
        return Unit.TOWER == unit_type

    def on_exit_timestep(self):
        # if not self.is_alive():
        #     self.farmworld.available_locs.add(self.location())
        if not self.is_alive():
            if self.cool_down_count == self.farmworld.tower_cooldown:
                self.farmworld.available_locs.add(self.location()) # when it first dies, add back
            self.cool_down_count -= 1
            if self.cool_down_count < 0:
                self._reset_tower()

    def get_encoding(self) -> np.ndarray:
        return np.array([0, 0, 1, 0, self._health/self._max_health, int(self.is_defended())])

        # return (np.array([Unit.TOWER, self._health, self.is_defended()])+1) * self.encoding_scaling # health, defended, unit_type==1, is_self = 0

    def get_img_rep(self) -> np.ndarray:
        scale = (self._max_health - max(min(self._health, self._max_health), 0)) / (self._max_health)
        if self.is_defended():
            idx = int(scale*(len(self.tower_health_indicators) - 1))
            base = Tower.tower_health_indicators[idx]
        else:
            idx = int(scale*(len(self.food_health_indicators) - 1))
            base = Tower.food_health_indicators[idx]
        return base
