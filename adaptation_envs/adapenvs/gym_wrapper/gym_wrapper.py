import gym

class GymWrapper():

    def __init__(self, config):
        self.env = gym.make(config['game'])
        self.context_size = config['context_size']
        self.context_sampler = config.get('context_sampler', 'l2')

        self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space
        self.eps_length = config.get('eps_length', 1_000)

        self.config = config
        self.num_agents = 1

        self.mode = config.get('mode', 'normal')
        self.noise_scale = config.get('noise_scale', 0.0)
        assert self.mode in ["normal", "cartpole_right", "cartpole_left"]
    
    def reset(self):
        self.tstep = 0
        return {1: self.env.reset()}

    def step(self, action_dict):
        obs, rew, done, infos = self.env.step(action_dict[1])

        self.tstep += 1
        if self.tstep > self.eps_length:
            done = True
        
        '''
        Here we have a bit of hard coding for specific gym environment ablations.
        '''
        if self.mode == "cartpole_left":
            rew = -obs[0] # x axis pos
        elif self.mode == "cartpole_right":
            rew = obs[0]

        return {1: obs}, {1: rew*0.1}, {1: done, "__all__": done}, {1: infos}
    
    def render(self):
        return self.env.render(mode="rgb_array")
        
    def close(self):
        self.env.close()