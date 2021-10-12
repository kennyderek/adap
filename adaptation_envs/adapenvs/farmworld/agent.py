from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from adapenvs.farmworld.unit import Unit
from adapenvs.farmworld.utilities import layer_images, get_resource_path
import random

import copy

class Agent(Unit):

    directions = np.array([[-1, 0], # UP (taken as y, x delta with 0,0 as top left)
                            [1, 0], # DOWN
                            [0, -1], # LEFT
                            [0, 1]]) # RIGHT

    rotation_directions = [270, 90, 0, 180]

    sword = np.array(Image.open(get_resource_path("sword.png")))
    pickaxe = np.array(Image.open(get_resource_path("pickaxe.png")))
    agent_health_indicators = [np.array(Image.open(get_resource_path("agent1.png"))),
                                np.array(Image.open(get_resource_path("agentd1 2.png"))),
                                np.array(Image.open(get_resource_path("agentd2 2.png"))),
                                np.array(Image.open(get_resource_path("agentd3 2.png"))),
                                np.array(Image.open(get_resource_path("agentd4.png")))]

    all_agent_imgs = []
    for r in rotation_directions:
        for mod in [None, sword, pickaxe]:
            for img in agent_health_indicators:
                if isinstance(mod, np.ndarray):
                    all_agent_imgs.append(np.array(Image.fromarray(layer_images([img, mod])).rotate(r)))
                else:
                    all_agent_imgs.append(np.array(Image.fromarray(img).rotate(r)))

    def __init__(self,
            farmworld,
            max_health : int,
            location:Union[None, Tuple[int, int]] = None,
            origin_time=0):
        self._max_health = max_health
        self._health = self._max_health

        # scale the encoding to be between 0 and 255 so we can use CNN policy structure
        self.encoding_scaling = 1/(np.array([Unit.ALL, self._max_health, 3])+1) * 255
        
        if location != None:
            self._location = np.array([*location])
            farmworld.available_locs.remove(location)
        else:
            self._location : np.ndarray = farmworld.pop_available_xy_coord()

        self.origin_time = origin_time

        self.reward = 0
        self.done = False
        self.orientation = 0
        self.cumulative_reward = 0

        self.farmworld = farmworld
        self.upper_bounds = np.array([farmworld.sidelength_y - 1, farmworld.sidelength_x - 1])
        self.lower_bounds = np.array([0, 0])

        self.mine = False
        self.attack = False
        self.error = False

        # metrics for specialization
        self.tower_damage = 0
        self.chicken_damage = 0
        # self.total_stats = 2

        # metrics to help logging
        self.chicken_attacks = 0
        self.tower_attacks = 0
        self.agent_attacks = 0
        self.lifetime = 0

    def get_square_infront(self) -> Tuple[int, int]:
        '''
        Returns a tuple of the square "in front" of the agent's orientation direction,
        even if that square is out-of-bounds (i.e. may contain negative values)
        '''
        return tuple(self._location + Agent.directions[self.orientation])

    def get_tower_damage(self):
        if not self.farmworld.use_specialization:
            return 1
        else:
            if self.chicken_damage == 0:
                self.tower_damage = 1
            else:
                self.error = True
            return self.tower_damage
            # self.tower_damage = min(self.farmworld.max_tower_health, self.tower_damage + 0.1)
            # self.chicken_damage = max(0, self.chicken_damage - 0.1)
            # return self.tower_damage

    def get_chicken_damage(self):
        if not self.farmworld.use_specialization:
            return 1
        else:
            if self.tower_damage == 0:
                self.chicken_damage = 1
            else:
                self.error = True
            return self.chicken_damage
            # self.chicken_damage = min(self.farmworld.max_chicken_health, self.chicken_damage + 0.1)
            # self.tower_damage = max(0, self.tower_damage - 0.1)
            # return self.chicken_damage

    def get_agent_damage(self):
        return 1

    def get_hay_damage(self):
        # return 1
        return 1
        # if not self.farmworld.use_specialization:
        #     return 1
        # else:
        #     if self.tower_damage == 0:
        #         self.chicken_damage = 1
        #     return self.chicken_damage

    @staticmethod
    def rotate(arr : np.ndarray, orientation):
        return np.array(Image.fromarray(arr).rotate(Agent.rotation_directions[orientation]))

    def location(self) -> Tuple[int, int]:
        return tuple(self._location)

    def get_obs(self, board, view_radius, sprite_dim):
        '''
        receives a board that is padded with view_radius zeroes on all sides
        returns: a slice of the board centered at agent_loc offset by padding, of L1 radius view_radius
        '''
        x = np.array([-view_radius, (view_radius+1)])
        y = np.copy(x)
        x = x + self._location[1] + view_radius
        y = y + self._location[0] + view_radius
        x, y = x * sprite_dim, y * sprite_dim
        agent_view_frame = board[y[0]:y[1],x[0]:x[1],:].view()
        # return np.transpose(agent_view_frame, (2, 0, 1))
        return agent_view_frame

    def get_health(self) -> int:
        return self._health

    def move(self, direction_idx : int):
        assert(direction_idx <= 4)
        self.orientation = direction_idx
        old_loc = self.location()
        self._location += Agent.directions[direction_idx]
        if self.location() not in self.farmworld.available_locs:
            self._location -= Agent.directions[direction_idx]
        else:
            self.farmworld.available_locs.add(old_loc)
            self.farmworld.available_locs.remove(self.location())
        self._location = np.clip(self._location, a_min=self.lower_bounds, a_max=self.upper_bounds)

    def modify_health(self, mod : int):
        self._health += mod
        self._health = min(self._max_health, self._health)
        if self._health <= 0:
            self.done = True
            # self.reward -= 0.9

    def is_alive(self):
        return not self.done

    def is_type(self, unit_type : int):
        return Unit.AGENT == unit_type

    def on_exit_timestep(self):
        
        self.modify_health(-1)
        self.cumulative_reward += self.reward
        
        if not self.is_alive():
            self.farmworld.available_locs.add(self.location())

    def get_encoding(self) -> np.ndarray:
        # print(np.array([0, 1, 0, 0, self._health/self._max_health, self.orientation/3], dtype=np.float))
        agent_encoding = [0, 1, 0, 0, self._health/self._max_health, self.orientation/3]

        return np.array(agent_encoding, dtype=np.float)

        # return (np.array([Unit.AGENT, self._health, self.orientation])+1) * self.encoding_scaling

    def get_img_rep(self) -> Tuple[np.ndarray, np.ndarray]:
        num_base_states = len(Agent.agent_health_indicators)
        num_non_rotated = len(Agent.all_agent_imgs) // 4
        scale = (self._max_health - max(min(self._health, self._max_health), 0)) / (self._max_health)
        idx = int(scale*(num_base_states-1))

        if self.attack:
            idx += num_base_states
        if self.mine:
            idx += num_base_states * 2
        idx = self.orientation * num_non_rotated + idx

        agent_img = copy.deepcopy(Agent.all_agent_imgs[idx])

        if self.tower_damage != 0 or self.chicken_damage != 0:
            box = np.zeros((9, 9, 4), dtype=np.uint8)
            box[0,...] = 1
            box[8,...] = 1
            box[:,0,:] = 1
            box[:,8,:] = 1
            box = box * np.array([255*self.chicken_damage, 255*self.tower_damage, 0, 1], dtype=np.uint8) # last one is the alpha

            if self.error:
                warn = np.zeros((9, 9, 4), dtype=np.uint8)
                warn[...,1] = 100
                # warn[....,3] = 1 # alpha
                agent_img += warn

            # print(agent_img)
            # print(box)
            agent_img += box
            # profile1 = np.array([0.25, 0.75, 0.25]) * 2
            # profile2 = np.array([0.75, 0.25, 0.75]) * 2
            # final = profile1 * self.tower_damage+ profile2 * self.chicken_damage
            # agent_img[...,0] = agent_img[...,0] * final[0]
            # agent_img[...,1] = agent_img[...,1] * final[1]
            # agent_img[...,2] = agent_img[...,2] * final[2]

        return agent_img