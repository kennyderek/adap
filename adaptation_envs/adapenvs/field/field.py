import math
import random
from copy import copy

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from ray.rllib.env import MultiAgentEnv
from collections import OrderedDict
from adapenvs.utils.context_sampler import SAMPLERS

class Field(MultiAgentEnv):
    """N-player discrete environment containing different food locations,
    once an agent one of the locations, the amount of food remaining there decreases by one. There
    are 4 food locations, and each of the food locations has n/4. If there is no remaining food,
    the agent will 'die'."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # NONE = 4

    def __init__(self, config = {}):
 
        # generic environment settings
        self.eps_len = config.get("eps_length", 30)
        self.num_agents = config.get("num_agents", 40)
        self.max_side = config.get("max_side", 10) # defines where the environment walls are

        # context settings
        self.context_size = config.get("context_size", 3)
        self.use_context = config.get("use_context", False)
        self.context_sampler = config.get("context_sampler", "l2")

        # Gym/RLLib settings
        assert config.get("max_env_actions") == 4, "should be 5 actions but is {}".format(config.get("max_env_actions"))
        self.action_space = Discrete(config.get("max_env_actions"))
        self.observation_space = Dict({
            "ctx": Box(-1, 1, (self.context_size,)),
            "obs": Box(0, 1, shape=((self.max_side*2+1)*2,))
        })

        # determines what reward agents get
        self.early_termination = config.get("early_termination", False)
            # if true, agent epsiodes end immediately upon hitting food, and recieve the reward of that food then.
            # Otherwise agents will continue until eps_len steps, and only recieve the reward for any food acquired at the end.

        # food location settings
        self.limited_food = config["limited_food"]
        self.food_even = config.get("food_even", 1)
        self.random_locs = config.get("random_locs", False)
        self.radius = config.get("radius", 6)
        if not self.random_locs:
            self.food_locations = [[self.radius, self.radius],
                                    [-self.radius, self.radius],
                                    [self.radius, -self.radius],
                                    [-self.radius, -self.radius]]
        else:
            self.food_locations = [[3, 4], # 7
                                    [-7, 2], # 9
                                    [-5, -5], # 9
                                    [6, -8]] # 14

        self.viewer = None
        self.reset()

    def reset(self):
        self.agent_contexts = SAMPLERS[self.context_sampler](ctx_size=self.context_size, num=self.num_agents)

        self.agent_positions = np.array([[0, 0] for _ in range(0, self.num_agents)])
        self.food_amounts = self._generate_random_food_amts()
        self.num_moves = 0
        self.who_found_food = np.array([0] * self.num_agents)
        self.agents = set(list(range(self.num_agents)))
        self.found = np.zeros((4,))

        return self._obs()

    def _generate_random_food_amts(self):
        if self.food_even == 1:
            if self.limited_food:
                return [math.floor(self.num_agents)/4]*4
            else:
                return [self.num_agents, self.num_agents, self.num_agents, self.num_agents]
        else:
            l = [self.num_agents//2, self.num_agents//4, self.num_agents//8, self.num_agents//8]
            random.shuffle(l)
            return copy(l)
    
    def _change_pos(self, agent_position, agent_num, move):
        if move == Field.UP:
            agent_position[agent_num] += np.array([0, -1])
        elif move == Field.DOWN:
            agent_position[agent_num] += np.array([0, 1])
        elif move == Field.LEFT:
            agent_position[agent_num] += np.array([-1, 0])
        elif move == Field.RIGHT:
            agent_position[agent_num] += np.array([1, 0])
        else:
            assert False, "action out of bounds: " + str(move)

    def _check_if_ate(self, agent_num):
        agent_loc = list(self.agent_positions[agent_num])
        if agent_loc in self.food_locations and self.who_found_food[agent_num] == 0:
            loc = self.food_locations.index(agent_loc)
            self.found[loc] = 1
            if self.food_amounts[loc] > 0:
                self.food_amounts[loc] -= 1
                self.who_found_food[agent_num] = 1

    def _obs(self):
        obs = {}
        for i in self.agents:
            loc = np.zeros(((self.max_side*2+1)*2,))
            loc[self.agent_positions[i][0] + self.max_side] = 1
            loc[self.agent_positions[i][1] + self.max_side + self.max_side*2+1] = 1
            # loc = self.agent_positions[i] / self.max_side
            if self.use_context:
                obs[i] = OrderedDict([
                    ("ctx", self.agent_contexts[i]), 
                    ("obs", loc)
                ])
            else:
                obs[i] = OrderedDict([("obs", loc)])
        return obs

    def _on_agent_end(self, agent_num):
        rew = 1 if (self.who_found_food[agent_num] == 1) else 0
        info = {"lifetime": self.num_moves}
        return rew, info

    def step(self, action_dict):
        self.num_moves += 1

        # process actions
        for agent_num, action in action_dict.items():
            self._change_pos(self.agent_positions, agent_num, action)
            self._check_if_ate(agent_num)
        self.agent_positions = np.clip(self.agent_positions, -self.max_side, self.max_side) # clip between

        # compute results
        obs, rewards, infos, dones = self._obs(), {}, {}, {"__all__": False}

        if self.num_moves >= self.eps_len or len(self.agents) == 0:
            dones["__all__"] = True
            if len(self.agents) > 0:
                a = np.random.choice(list(self.agents))
                infos[a] = {"targets_found": np.sum(self.found)}
        
        for agent_num in copy(self.agents):
            if (self.who_found_food[agent_num] == 1 and self.early_termination) or dones["__all__"]:
                rew, info = self._on_agent_end(agent_num)
                if agent_num in infos:
                    infos[agent_num]["lifetime"] = info["lifetime"]
                else:
                    infos[agent_num] = info
                rewards[agent_num] = rew
                dones[agent_num] = True
                self.agents.remove(agent_num)
            else:
                rewards[agent_num] = 0
                dones[agent_num] = False

        return obs, rewards, dones, infos

