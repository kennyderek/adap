import json
import random
from typing import Dict, List, Set, Tuple
from copy import copy
import copy

from gym import spaces
import numpy as np
from adapenvs.farmworld.agent import Agent
from adapenvs.farmworld.tower import Tower
from adapenvs.farmworld.chicken import Chicken
from adapenvs.farmworld.wall import Wall

from adapenvs.farmworld.unit import Unit
from adapenvs.farmworld.utilities import (get_image_and_mask,
                                              get_resource_path, layer_images,
                                              layer_two_images_fast, get_resource_path_string)
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from PIL import Image, ImageDraw, ImageFont

from collections import OrderedDict

class Farmworld():

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    MINE = 4
    ATTACK = 5

    base_img1 = np.array(Image.open(get_resource_path("ground1.png")))
    base_img2 = np.array(Image.open(get_resource_path("ground2.png")))

    sprite_width = 9
    encoding_width = 1

    def __init__(self, config):
        self.config = config

        # initialize constant variables
        self.max_agent_health = config.get("max_agent_health", 20)
        self.max_tower_health = config.get("max_tower_health", 1)
        self.max_chicken_health = config.get("max_chicken_health", 1)
        self.agent_view_range = config.get("agent_view_range", 2)
        self.agent_payoff = config.get('agent_payoff', 0)
        self.tower_payoff = config.get('tower_payoff', 5)
        self.chicken_payoff = config.get('chicken_payoff', 5)
        self.ava_damage = config.get('ava_damage', 0) # agent vs. agent damage
        self.use_specialization = config.get("use_specialization", False)
        self.use_pixels = config.get("use_pixels", False)
        self.map = config.get("map", [])
        
        self.action_space = Discrete(config.get("max_env_actions", 6))
        assert config.get("max_env_actions", 6) == 6, "should be 6 actions but is {}".format(config.get("max_env_actions", 6))

        self.eps_length = config.get("eps_length", 100)
        
        if self.map != []:
            self.sidelength_y = len(self.map)
            self.sidelength_x = len(self.map[0])
            num_agents = 0
            num_chickens = 0
            num_towers = 0
            num_walls = 0
            for row in range(len(self.map)):
                for col in range(len(self.map[row])):
                    if self.map[row][col] == "c":
                        num_chickens += 1
                    elif self.map[row][col] == "a":
                        num_agents += 1
                    elif self.map[row][col] == "t":
                        num_towers += 1
                    elif self.map[row][col] == "w":
                        num_walls += 1
            self.original_num_agents = num_agents
            self.num_agents = self.original_num_agents
            self.num_chickens = num_chickens
            self.num_towers = num_towers
            self.num_walls = num_walls
        else:
            self.original_num_agents = config.get("num_agents", 2)
            self.num_agents = self.original_num_agents
            self.num_chickens = config.get("num_chickens", 0)
            self.sidelength_x = config.get("sidelength_x", 5)
            self.sidelength_y = config.get("sidelength_y", 5)
            self.num_towers = config.get("num_towers", 0)

        self.tower_cooldown = config.get('tower_cooldown', 0)
        self.chicken_cooldown = config.get('chicken_cooldown', 0)

        self.pixel_viz = False
        if self.use_pixels:
            self.sprite_width = Farmworld.sprite_width # the width/height of a single unit in Farmworld (i.e. a 9x9 png), this could be 1, and have different depth
        else:
            self.sprite_width = Farmworld.encoding_width
        self.map_pixel_size = (2*(self.agent_view_range)+1) * self.sprite_width

        if self.use_pixels:
            if config.get("channels_first", False):
                self.transpose_out = [2, 0, 1]
                self.observation_space = Box(0, 255, shape=[3, self.map_pixel_size, self.map_pixel_size], dtype=np.uint8)
            else:
                self.transpose_out = [0, 1, 2]
                self.observation_space = Box(0, 255, shape=[self.map_pixel_size, self.map_pixel_size, 3], dtype=np.uint8)
        else:
            cell_encoding_dim = 6
            
            num_cells = (self.agent_view_range*2+1) ** 2
            obs_dim = num_cells * cell_encoding_dim
            self.observation_space = Box(0, 1, shape=(obs_dim,), dtype=np.float)


        self.reset()

    def pop_available_xy_coord(self) -> np.ndarray:
        '''
        Spawn a random 2 dim nd array in range of [0, sidelength]^2
        '''
        loc = random.choice(tuple(self.available_locs))
        self.available_locs.remove(loc)
        return np.array(loc)

    def assign_random(self):
        for agent_id in range(1, self.num_agents+1):
            self.agents[agent_id] = Agent(self,
                    max_health=self.max_agent_health,
                    location=None,
                    origin_time=0)
        
        for _ in range(self.num_towers):
            self.towers.append(Tower(self, self.max_tower_health))

        for _ in range(self.num_chickens):
            self.chickens.append(Chicken(self, self.max_chicken_health))

    def assign_by_map(self):
        agent_id = 1
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                if self.map[row][col] == "c":
                    self.chickens.append(Chicken(self, self.max_chicken_health, loc=(row, col)))
                elif self.map[row][col] == "a":
                    self.agents[agent_id] = Agent(self,
                            max_health=self.max_agent_health,
                            location=(row, col),
                            origin_time=0)
                    agent_id += 1
                elif self.map[row][col] == "t":
                    self.towers.append(Tower(self, self.max_tower_health, loc=(row, col)))
                elif self.map[row][col] == "w":
                    self.walls.append(Wall(self, loc=(row, col)))

    def reset(self):
        self.available_locs : Set[Tuple[int, int]] = set([(j, i) for i in range(self.sidelength_x) for j in range(self.sidelength_y)])
        # print(self.available_locs)
        self.step_num = 0
        self.num_agents = self.original_num_agents
        self.agents : Dict[int, Agent] = {}
        self.towers : List[Tower] = []
        self.chickens : List[Chicken] = []
        self.walls : List[Wall] = []
        self.total_num_agents = self.num_agents
        self.num_errors = 0 # number of errors in enforced specialization exp

        if self.map == []:
            self.assign_random()
        else:
            self.assign_by_map()
        self.dead_agents = [] # for keeping track of agents that are gone

        self.base_map_img = np.zeros((self.sidelength_y * Farmworld.sprite_width, self.sidelength_x * Farmworld.sprite_width, 4), dtype=np.uint8)
        # draw the grassy texture
        for i in range(self.sidelength_x):
            for j in range(self.sidelength_y):
                x1, x2 = i * Farmworld.sprite_width, (i + 1) * Farmworld.sprite_width
                y1, y2 = j * Farmworld.sprite_width, (j + 1) * Farmworld.sprite_width
                temp = random.choice([self.base_img1, self.base_img2])
                self.base_map_img[y1:y2,x1:x2,:] = temp
        self.last_img = copy.deepcopy(self.base_map_img)

        self.base_map_enc = np.zeros((self.sidelength_y * self.sprite_width, self.sidelength_x * self.sprite_width, 6), dtype=np.float)

        return self._obs()

    def _compute_target_entropy(self, action_1, action_2):
        ent = None
        total = action_1 + action_2
        if total != 0:
            p1 = action_1 / total
            p2 = action_2 / total
            ent = (p1 * np.log2(p1) if p1 != 0 else 0) + (p2 * np.log2(p2) if p2 != 0 else 0)
        return ent

    def _on_agent_end(self, agent_id):
        '''
        compute info stats for this agent
        '''
        agent = self.agents[agent_id]
        info = {}
        # compute the entropy of what this agent attacked, during its lifespan
        attacktropy = self._compute_target_entropy(agent.chicken_attacks, agent.tower_attacks)
        if attacktropy != None:
            info['c_t_attacktropy'] = -attacktropy
        attacktropy = self._compute_target_entropy(agent.chicken_attacks + agent.tower_attacks, agent.agent_attacks)
        if attacktropy != None:
            info['ct_a_attacktropy'] = -attacktropy

        info['avt'] = agent.tower_attacks
        info['avc'] = agent.chicken_attacks
        info['ava'] = agent.agent_attacks
        info['lifetime'] = self.step_num - agent.origin_time
        return info

    def _obs(self) -> Dict[int, np.ndarray]:
        padding = self.sprite_width * self.agent_view_range
        if self.use_pixels:
            board = self.get_world_map(padding=padding)
        else:
            board = self.get_encoded_map(padding=padding)

        transform = lambda obs: obs.flatten()
        if self.use_pixels:
            transform = lambda obs: obs.transpose(self.transpose_out)
        else:
            transform = lambda obs: obs.flatten()
        
        obs = {}
        for agent_id in self.agents:
            transform = lambda obs: obs.flatten()
            agent_obs = transform(self.agents[agent_id].get_obs(
                                        board, self.agent_view_range,
                                        self.sprite_width))
            obs[agent_id] = agent_obs
        return obs

    def step(self, actions : Dict[int, int]):
        # reset agent states for this step
        for agent_id, agent in self.agents.items():
            agent.reward = 0.1 # positive reward for surviving longer
            agent.mine = False
            agent.attack = False
            agent.error = False

        action_queue = list(actions.items())
        random.shuffle(action_queue)

        # build a map of current locations
        all_units : List[Unit] = list(self.agents.values()) + self.towers + self.chickens + self.walls
        loc_map : Dict[Tuple[int, int], Unit] = {unit.location():[] for unit in all_units}
        for unit in all_units:
            loc_map[unit.location()].append(unit)

        # process all MINE/ATTACK actions
        for agent_id, action in action_queue:
            if agent_id in self.agents and action in [Farmworld.MINE, Farmworld.ATTACK]:
                agent : Agent = self.agents[agent_id]
                square_infront = agent.get_square_infront()
                agent.attack = (action == Farmworld.ATTACK)
                agent.mine = (action == Farmworld.MINE)
                units_at_target_location : List[Unit] = loc_map.get(square_infront, [])
                for other in units_at_target_location:
                    if other.is_alive() and agent.is_alive():
                        if action == Farmworld.ATTACK:
                            if other.is_type(Unit.AGENT):
                                agent.agent_attacks += 1
                                other.modify_health(-agent.get_agent_damage()*self.ava_damage)
                                if not other.is_alive():
                                    agent.modify_health(self.agent_payoff)
                            elif other.is_type(Unit.CHICKEN):
                                agent.chicken_attacks += 1
                                other.modify_health(-agent.get_chicken_damage())
                                if not other.is_alive():
                                    agent.modify_health(self.chicken_payoff)
                            elif other.is_type(Unit.TOWER) and other.is_defended():
                                agent.tower_attacks += 1
                                other.modify_health(-agent.get_tower_damage())
                            elif other.is_type(Unit.TOWER):
                                agent.tower_attacks += 1
                        elif action == Farmworld.MINE:
                            if other.is_type(Unit.TOWER) and not other.is_defended():
                                agent.tower_attacks += 1
                                other.modify_health(-agent.get_hay_damage())
                                if not other.is_alive():
                                    agent.modify_health(self.tower_payoff)
                            elif other.is_type(Unit.CHICKEN):
                                agent.chicken_attacks += 1
                            elif other.is_type(Unit.AGENT):
                                agent.agent_attacks += 1
                            elif other.is_type(Unit.TOWER):
                                agent.tower_attacks += 1
                            
        # process all MOVE actions
        for agent_id, action in action_queue:
            # assert isinstance(action, int), type(action)
            if 0 <= action < Farmworld.MINE and agent_id in self.agents:
                self.agents[agent_id].move(action)

        # run on-exit for all agents (which frees of the location of any non-alive units)
        for unit in all_units:
            unit.on_exit_timestep()

        #
        # build obs, rewards, dones, infos
        # we log info (per agent) when an agent dies, or when the game ends
        #
        rewards = {agent_id:self.agents[agent_id].reward for agent_id in self.agents}
        dones = {}
        infos = {}
        obs = self._obs()

        # clear remove agents & chickens that have died
        all_agents : List[Tuple[int, Agent]]= list(self.agents.items())
        for agent_id, agent in all_agents:
            if agent.error:
                self.num_errors += 1
            if not agent.is_alive():
                infos[agent_id] = self._on_agent_end(agent_id)
                dones[agent_id] = True
                agent.lifetime = self.step_num - agent.origin_time
                self.dead_agents.append(agent)
                del self.agents[agent_id]
            else:
                dones[agent_id] = False

        dones["__all__"] = self.step_num >= self.eps_length or len(self.agents) == 0
        if dones["__all__"]:
            for agent_id, agent in self.agents.items():
                infos[agent_id] = self._on_agent_end(agent_id)
                dones[agent_id] = True # set to done when episode is over?
                agent.lifetime = self.step_num - agent.origin_time
                self.dead_agents.append(agent)

        self.step_num += 1
        return obs, rewards, dones, infos

    def get_encoded_map(self, padding=0) -> np.ndarray:
        all_units : List[Unit] = list(self.agents.values()) + self.towers + self.chickens + self.walls

        padded_sidelength = np.array(self.base_map_enc.shape) + 2 * padding * self.sprite_width
        offset_img = np.zeros((padded_sidelength[0], padded_sidelength[1], 6), dtype=np.float)
        
        offset_img[padding:padded_sidelength[0]+padding,padding:padded_sidelength[1]+padding,0] = 1

        for unit in [u for u in all_units if u.is_alive()]:
            l = unit.location()
            x1, x2 = l[0] * Farmworld.encoding_width + padding, (l[0] + 1) * Farmworld.encoding_width + padding
            y1, y2 = l[1] * Farmworld.encoding_width + padding, (l[1] + 1) * Farmworld.encoding_width + padding
            unit_encoding = unit.get_encoding().view()
            offset_img[x1:x2,y1:y2,:len(unit_encoding)] = unit_encoding

        return offset_img.view()

    def get_world_map(self, padding=0) -> np.ndarray:
        all_units : List[Unit] = list(self.agents.values()) + self.towers + self.chickens + self.walls

        padded_sidelength = np.array(self.base_map_img.shape) + 2 * padding
        offset_img = np.zeros((padded_sidelength[0], padded_sidelength[1], 4), dtype=np.uint8)
        offset_img[padding:padded_sidelength[0]-padding,padding:padded_sidelength[1]-padding,:] = copy.deepcopy(self.last_img)

        for unit in [u for u in all_units if u.is_alive()]:
            l = unit.location()
            x1, x2 = l[0] * Farmworld.sprite_width + padding, (l[0] + 1) * Farmworld.sprite_width + padding
            y1, y2 = l[1] * Farmworld.sprite_width + padding, (l[1] + 1) * Farmworld.sprite_width + padding
            layer_two_images_fast(offset_img[x1:x2,y1:y2,:], unit.get_img_rep().view())

        offset_img = offset_img.astype(np.uint8)

        # size = 500
        # scale_y = int(size * 5 / 5)
        # scale_x = int(size * 4 / 5)
        # img = Image.fromarray(offset_img[...,0:3].view()).resize((scale_y, scale_x), Image.NEAREST)

        # draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(str(get_resource_path_string("arial.ttf")), 35)
        # draw.text((4, 4),"Number of agent `blunders':" + str(self.num_errors), (0,0,255), font=font)

        return offset_img[...,0:3]

    def render(self):
        return self.get_world_map()