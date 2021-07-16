import math
import random
from copy import copy

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from ray.rllib.env import MultiAgentEnv
from collections import OrderedDict

from adapenvs.utils.context_sampler import SAMPLERS, TRANSFORMS

from PIL import Image, ImageDraw, ImageFont


class MarkovSoccer(MultiAgentEnv):
    """
    """

    RIGHT = 0
    LEFT = 1
    DOWN = 2
    UP = 3
    STAND = 4

    A = 0
    B = 1

    MOVE_SET = np.array([[1, 0], # RIGHT
                         [-1, 0],# LEFT
                         [0, 1], # DOWN
                         [0, -1],# UP
                         [0, 0]  # STAND
                        ])

    A_GOALS = np.array([[5, 1], [5, 2]])
    B_GOALS = np.array([[-1, 1], [-1, 2]])

    # for visualization
    A_IMG = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    B_IMG = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])  
    SQUARE = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]])

    def __init__(self, config = {}):
        self.context_size = config.get("context_size", 4)
        self.use_context = config.get("use_context", True)
        self.context_sampler = config.get("context_sampler", "l2")

        self.action_space = Discrete(config.get("max_env_actions"))
        assert config.get("max_env_actions") == 5

        # we provide the 4 row x 5 col board,
        # where A is -1 and B player is 1, everything else is 0
        # A always starts on left side, and B starts on right side
        # possession of ball is indicated by an additional byte (-1, 1)

        # self.observation_space = Dict({
        #     "ctx": Box(-1, 1, (self.context_size,)),
        #     "obs": Box(-1, 1, (4*5+1,)) # board shape, -1/1 for possession of ball
        # })
        self.observation_space = Dict({
            "ctx": Box(-1, 1, (self.context_size,)),
            "obs": Box(-1, 1, (11,))
        })
        self.A_goals = np.array([[5, 1], [5, 2]])
        self.B_goals = np.array([[-1, 1], [-1, 2]])
        self.mode = config.get("mode", "normal")

        self.reset()

    def reset(self):

        self.contexts = SAMPLERS[self.context_sampler](ctx_size=self.context_size, num=2)
        self.pos_B = np.random.randint([3, 1],[5,3]) # sample x \in [3, 5) and y in [1, 3)

        self.dones = {"__all__": False}
        self.rewards = {MarkovSoccer.A:0, MarkovSoccer.B:0}

        if self.mode == "normal":
            # in <x, y>
            self.pos_A = np.random.randint([0, 1],[2,3]) # sample x \in [0, 2) and y in [1, 3)
            self.possesion = random.choice([MarkovSoccer.A, MarkovSoccer.B])
        elif self.mode == "stand":
            self.pos_A = np.array([0, 1])
            self.possesion = MarkovSoccer.B
        elif self.mode == "straight":
            self.pos_A = np.array([0, 1])
            self.possesion = MarkovSoccer.A
        elif self.mode == "updown1":
            self.pos_A = np.array([1, 1])
            self.possesion = MarkovSoccer.B
        elif self.mode == "updown0":
            self.pos_A = np.array([0, 1])
            self.possesion = MarkovSoccer.B
        elif self.mode == "random":
            self.pos_A = np.random.randint([0, 1],[2,3]) # sample x \in [0, 2) and y in [1, 3)
            self.possesion = random.choice([MarkovSoccer.A, MarkovSoccer.B])
        elif self.mode == "planned" or self.mode == "epsilon-planned":
            # same init. conditions as normal
            self.pos_A = np.random.randint([0, 1],[2,3]) # sample x \in [0, 2) and y in [1, 3)
            self.possesion = random.choice([MarkovSoccer.A, MarkovSoccer.B])
        else:
            assert False, "shouldn't get here"

        self.winner = None

        return self._obs()

    def _obs(self):
        obs = {}

        obs_A = np.concatenate([-(self.pos_A - self.A_GOALS).flatten() / 5,
                                -(self.pos_B - self.B_GOALS).flatten() / 5,
                                -(self.pos_A - self.pos_B) / 5,
                                np.array([int(self.possesion == MarkovSoccer.A)])])
                                
        obs_B = np.concatenate([(self.pos_B - self.B_GOALS).flatten() / 5,
                                (self.pos_A - self.A_GOALS).flatten() / 5,
                                (self.pos_B - self.pos_A) / 5,
                                np.array([int(self.possesion == MarkovSoccer.B)])])

        obs[0] = OrderedDict([
            ("ctx", self.contexts[0]),
            ("obs", obs_A) # player A will get a reflected view
        ])
        obs[1] = OrderedDict([
            ("ctx", self.contexts[1]),
            ("obs", obs_B)
        ])

        if self.mode != "normal":
            del obs[0] # delete player A observation

        return obs

    def get_A_context(self):
        return self.contexts[0]
    
    def get_B_context(self):
        return self.contexts[1]

    def set_A_context(self, ctx):
        self.contexts[0] = ctx
    
    def set_B_context(self, ctx):
        self.contexts[1] = ctx

    def _flip_action(self, action):
        if action == MarkovSoccer.LEFT:
            action = MarkovSoccer.RIGHT
        elif action == MarkovSoccer.RIGHT:
            action = MarkovSoccer.LEFT
        elif action == MarkovSoccer.UP:
            action = MarkovSoccer.DOWN
        elif action == MarkovSoccer.DOWN:
            action = MarkovSoccer.UP
        return action

    def _move_player_A(self, action):
        # A's view is reflected, so we need to swap the L and R moves too (up and down stay the same!)
        action = self._flip_action(action)
        
        self.pos_A += MarkovSoccer.MOVE_SET[action]
        if (self.pos_A == self.pos_B).all():
            self.pos_A -= MarkovSoccer.MOVE_SET[action]
            self.possesion = MarkovSoccer.B
        if (self.pos_A == MarkovSoccer.A_GOALS).all(axis=-1).any() and self.possesion == MarkovSoccer.A:
            self.winner = MarkovSoccer.A
        self.pos_A = np.clip(self.pos_A, [0, 0], [4, 3]) # clip x to 4, y to 3

    def _move_player_B(self, action):
        self.pos_B += MarkovSoccer.MOVE_SET[action]
        if (self.pos_B == self.pos_A).all():
            self.pos_B -= MarkovSoccer.MOVE_SET[action]
            self.possesion = MarkovSoccer.A
        if (self.pos_B == MarkovSoccer.B_GOALS).all(axis=-1).any() and self.possesion == MarkovSoccer.B:
            self.winner = MarkovSoccer.B
        self.pos_B = np.clip(self.pos_B, [0, 0], [4, 3]) # clip x to 4, y to 3

    def _get_planned_action(self):
        pos_diff = (self.pos_A - self.pos_B)
        action = None
        diff_y = pos_diff[1]
        diff_x = pos_diff[0]
        if self.possesion == MarkovSoccer.B:
            # if we are defending, always be blocking B, and between B and the goal
            if diff_x > 0:
                # A is to the right of B
                action = MarkovSoccer.LEFT
            elif diff_y < 0:
                # A is above B
                action = MarkovSoccer.DOWN
            elif diff_y > 0:
                # A is below B
                action = MarkovSoccer.UP
            elif diff_x < -1:
                # A is too far to the left of B
                action = MarkovSoccer.RIGHT
            else:
                action = MarkovSoccer.STAND
        else:
            # we are on the offense, stay in the two lanes directly in front of the goal,
            #   and go forward if B is not in front of us. Otherwise, swap lanes
            if self.pos_A[1] == 0: # top row, move down
                action = MarkovSoccer.DOWN
            elif self.pos_A[1] == 3: # bottom row, move up
                action = MarkovSoccer.UP
            elif diff_x == -1 and diff_y == 0:
                # B is directly in front of us, swap lanes
                if self.pos_A[1] == 1:
                    action = MarkovSoccer.DOWN
                elif self.pos_A[1] == 2:
                    action = MarkovSoccer.UP
                else:
                    assert False, "shouldn't get here" + str(self.pos_A)
            else:
                # path is clear, move forward!
                action = MarkovSoccer.RIGHT
        
        return self._flip_action(action)


    def step(self, action_dict):
        if self.mode != "normal":
            if self.mode == "straight":
                # hard coded A always moves left (which is actually right, since their moves are reversed)
                action_dict[0] = MarkovSoccer.LEFT
            elif self.mode == "stand":
                # hard coded A always defends/stands!
                action_dict[0] = MarkovSoccer.STAND
            elif self.mode == "updown1" or self.mode == "updown0":
                # move up and down
                if self.pos_A[1] == 2:
                    action_dict[0] = MarkovSoccer.DOWN
                else:
                    action_dict[0] = MarkovSoccer.UP
            elif self.mode == "random":
                action_dict[0] = random.randint(0, 4)
            elif self.mode == "planned":
                action_dict[0] = self._get_planned_action()
            elif self.mode == "epsilon-planned":
                if random.random() < 0.3:
                    action_dict[0] = random.randint(0, 4)
                else:
                    action_dict[0] = self._get_planned_action()
            else:
                assert False, "shouldn't reach here"

        if random.random() < 0.5:
            self._move_player_A(action_dict[0])
            self._move_player_B(action_dict[1])
        else:
            self._move_player_B(action_dict[1])
            self._move_player_A(action_dict[0])

        self.rewards = {MarkovSoccer.A: 0, MarkovSoccer.B: 0}
        self.dones = {MarkovSoccer.A: False, MarkovSoccer.B: False, "__all__": False}
        infos = {}

        if random.random() < 0.05 or self.winner != None:
            if self.winner == MarkovSoccer.A:
                self.rewards[MarkovSoccer.A] = 1
                self.rewards[MarkovSoccer.B] = -1
            elif self.winner == MarkovSoccer.B:
                self.rewards[MarkovSoccer.A] = -1
                self.rewards[MarkovSoccer.B] = 1
            self.dones = {MarkovSoccer.A: True, MarkovSoccer.B: True, "__all__": True}
            info_dict = {"A": int(self.winner == MarkovSoccer.A), "B": int(self.winner == MarkovSoccer.B), "Draw": int(self.winner == None)}
            infos[MarkovSoccer.A] = info_dict
            infos[MarkovSoccer.B] = info_dict

        if self.mode != 'normal':
            del self.rewards[0]
            del self.dones[0]
            if 0 in infos:
                del infos[0]

        return self._obs(), self.rewards, self.dones, infos

    def get_game_board(self):
        board = np.zeros((4*9, 5*9))
        for y in range(0, 4):
            for x in range(0, 5):
                board[y*9:(y+1)*9, x*9:(x+1)*9] += MarkovSoccer.SQUARE
        
        x = self.pos_A[0]
        y = self.pos_A[1]
        board[y*9:(y+1)*9, x*9:(x+1)*9] += MarkovSoccer.A_IMG

        x = self.pos_B[0]
        y = self.pos_B[1]
        board[y*9:(y+1)*9, x*9:(x+1)*9] += MarkovSoccer.B_IMG

        if self.possesion == MarkovSoccer.A:
            possesion_square = self.pos_A
        else:
            possesion_square = self.pos_B
        
        x = possesion_square[0]
        y = possesion_square[1]
        board[y*9:(y+1)*9, x*9:(x+1)*9] += 0.75
        np.clip(board[y*9:(y+1)*9, x*9:(x+1)*9], 0, 1, out=board[y*9:(y+1)*9, x*9:(x+1)*9])

        return board
    
    def render(self):
        '''
        Returns RGB representation of the board
        '''
        size = 500
        scale_y = int(size * 5 / 5)
        scale_x = int(size * 4 / 5)
        img = Image.fromarray(self.get_game_board() * 255).resize((scale_y, scale_x), Image.NEAREST)

        if self.dones["__all__"]:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("utils/arial.ttf", 120)

            if self.rewards[0] == 1:
                draw.text((4, 4),"A Wins!", (0,255,0), font=font)
            elif self.rewards[1] == 1:
                draw.text((4, 4),"B Wins!", (255,0,0), font=font)
            else:
                draw.text((4, 4),"Draw!", (0,0,255), font=font)

        return np.array(img)