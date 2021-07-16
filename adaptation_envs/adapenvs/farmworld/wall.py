from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from adapenvs.farmworld.unit import Unit
from adapenvs.farmworld.utilities import layer_images, get_resource_path
import random

class Wall(Unit):

    wall_img = np.array(Image.open(get_resource_path("fence.png")))

    def __init__(self, farmworld, loc:Union[None, Tuple[int, int]] = None):
        if loc != None:
            self._location = tuple(np.array([*loc]))
            farmworld.available_locs.remove(loc)
        else:
            self._location : np.ndarray = farmworld.pop_available_xy_coord()

    def location(self) -> Tuple[int, int]:
        return self._location

    def get_health(self) -> int:
        return 1

    def modify_health(self, mod : int):
        pass

    def is_alive(self):
        return True

    def is_type(self, unit_type : int):
        return Unit.WALL == unit_type

    def on_exit_timestep(self):
        pass

    def get_encoding(self) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 0]) # we treat walls as we do the empty void of blackness that is out there!

    def get_img_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        return Wall.wall_img