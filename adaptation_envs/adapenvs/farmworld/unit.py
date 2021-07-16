from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np


class Unit(ABC):
    # represents a unit, i.e. an agent, or a tower, or a food depot, etc

    ENCODING_LENGTH = 4
    AGENT = 0
    TOWER = 1
    CHICKEN = 2
    GROUND = 3
    GRENADE = 4
    CAKE = 5
    WALL = 6
    ALL = 6 # this is the max number of units, to be updated as we add units

    @abstractmethod
    def location(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def is_type(self, unit_type : int) -> int:
        pass

    @abstractmethod
    def modify_health(self, mod : int):
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        pass

    @abstractmethod
    def on_exit_timestep(self):
        '''
        A method that will be called on all Unit types before an observation is returned

        info: a dictionary containing information that the method needs to know
        '''
        pass

    @abstractmethod
    def get_encoding(self) -> np.ndarray:
        return np.array([0, 0, 0])

    @abstractmethod
    def get_img_rep(self) -> np.ndarray:
        pass

