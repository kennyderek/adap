import gym
from collections import OrderedDict
from gym import spaces
from adapenvs.utils.context_sampler import TRANSFORMS, SAMPLERS
import numpy as np
import random

from adapenvs.farmworld.farmworld import Farmworld


class LatentWrapper():

    def __init__(self, env):
        self.env = env
        self.latent_size = env.config.get('context_size', 3)
        self.num_agents = env.num_agents

        self.default_latent_sampler = env.config.get('latent_sampler', 'l2')
        default_sampler_fn = lambda : SAMPLERS[self.default_latent_sampler](ctx_size=self.latent_size, num=1)[0]
        self.agent_id_to_sampler_fn = dict((i, default_sampler_fn) for i in range(1, env.num_agents + 1))

        self.observation_space = spaces.Dict({
            "ctx": spaces.Box(-1, 1, (self.latent_size,)),
            "obs": env.observation_space
        })
        self.action_space = env.action_space

        self.agent_latents = None

    def _wrap_obs(self, obs, i):
        return OrderedDict([('ctx', self.agent_latents[i]), ('obs', obs)])

    def _obs(self, all_obs):
        return dict((i, self._wrap_obs(obs, i)) for (i, obs) in all_obs.items())

    def reset(self):
        self.agent_latents = dict([(i, self.agent_id_to_sampler_fn[i]()) for i in range(1, self.num_agents + 1)])

        all_obs = self.env.reset()
        return self._obs(all_obs)

    def step(self, action_dict):
        all_obs, all_rew, all_dones, all_infos = self.env.step(action_dict)

        return self._obs(all_obs), all_rew, all_dones, all_infos

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def set_sampler_fn(self, agent_id, fn):
        self.agent_id_to_sampler_fn[agent_id] = fn

class AgentLatentOptimizer():
    def __init__(self, latent_size, default_sampler, steps):
        self.latent_size = latent_size
        self.default_sampler = default_sampler
        self.default_sampler_fn = lambda : SAMPLERS[default_sampler](ctx_size=self.latent_size, num=1)[0]

        self.evolve_steps=steps
        self.random_thres=0.6

        # after [finetuning_ratio * evolve_steps] steps, we only pop from the top 10 latents
        self.finetuning_ratio=3/4

        self.replication_thres=0.3
        self.mutation_thres=0.75

        assert self.replication_thres < self.mutation_thres

        self.best = []
        self.estep = 0

        self.last_latent = None

    def sampler_fn(self):
        if (random.random() < self.random_thres or len(self.best) < 10) and self.estep < int(self.evolve_steps * self.finetuning_ratio):
                z = self.default_sampler_fn()
        else:
            r = random.random()
            if r < self.replication_thres and self.estep < int(self.evolve_steps * self.finetuning_ratio):
                z = random.choice(self.best[0:10])[0]
            elif r < self.mutation_thres and self.estep < int(self.evolve_steps * self.finetuning_ratio):
                choice = random.choice(self.best[0:10])[0]
                z = choice + np.random.random(choice.shape) / 20
                z = TRANSFORMS[self.default_sampler](z)
            else:
                z = self.best.pop(random.randint(0, min(10, len(self.best)-1)))[0]
        
        self.last_latent = z
        return z

    def get_best_fn(self):
        return lambda : random.choice(self.best[0:5])[0]

    def update(self, reward):
        self.best.append((self.last_latent, reward))
        self.best.sort(key = lambda x: x[1], reverse=True)

