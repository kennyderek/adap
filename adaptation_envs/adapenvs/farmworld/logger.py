
class LoggingMetrics():
    
    def __init__(self, log_file):
        self.log_file = log_file

        self.agent_v_agent = 0
        self.agent_v_tower = 0
        self.agent_v_chicken = 0
        self.merges = 0
        self.gives = 0
        self.min_lifetime = None
        self.max_lifetime = None
        self.mean_lifetime = None
        self.agent_contexts = []
        self.agent_entropies = []
        self.eps_len = 0

        self.rewards = []
        self.lifetimes = []
        self.meta = []

    def write(self, vars_to_avoid = ["lifetimes", "log_file", "rewards"]):
        if self.log_file != "":
            with open(self.log_file, 'a') as f:
                data = {k: v for k, v in self.__dict__.items() if k not in vars_to_avoid}
                f.write(str(data) + "\n")

    def set_vars(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
