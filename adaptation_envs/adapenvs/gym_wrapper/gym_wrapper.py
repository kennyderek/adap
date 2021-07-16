import gym
from ray.rllib.env import MultiAgentEnv
from collections import OrderedDict
from gym import spaces
from adapenvs.utils.context_sampler import SAMPLERS
import numpy as np
import math

class GymWrapper(MultiAgentEnv):

    def __init__(self, config):
        self.env = gym.make(config['game'])
        self.context_size = config['context_size']
        self.context_sampler = config.get('context_sampler', 'l2')

        self.use_time = config.get('use_time', False)

        obs_space = self.env.observation_space

        self.observation_space = spaces.Dict({
            "ctx": spaces.Box(-1, 1, (self.context_size,)),
            "obs": obs_space
        })
        self.action_space = self.env.action_space
        self.max_episode_len = config.get('max_episode_len', 1_000)

        self.mode = config.get('mode', 'normal')
        self.noise_scale = config.get('noise_scale', 0.0)
        assert self.mode in ["normal", "cartpole_right", "cartpole_left"]

    def _wrap_obs(self, obs):
        return OrderedDict([('ctx', self.context), ('obs', obs)])

    def reset(self, latent_z=None):
        self.tstep = 0
        obs = self.env.reset()
        
        if isinstance(latent_z, np.ndarray):
            self.context = latent_z
        else:
            self.context = SAMPLERS[self.context_sampler](ctx_size=self.context_size, num=1)[0]

        return {0: self._wrap_obs(obs)}

    def step(self, action_dict):
        obs, rew, done, infos = self.env.step(action_dict[0])

        self.tstep += 1
        if self.tstep > self.max_episode_len:
            done = True
        
        '''
        Here we have a bit of hard coding for specific gym environment ablations.
        '''
        if self.mode == "cartpole_left":
            rew = -obs[0] # x axis pos
        elif self.mode == "cartpole_right":
            rew = obs[0]

        return {0: self._wrap_obs(obs)}, {0: rew*0.1}, {0: done, "__all__": done}, {0: infos}
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
        
    def close(self):
        self.env.close()

    def set_context(self, context):
        self.context = context
    
    def get_context(self):
        return self.context