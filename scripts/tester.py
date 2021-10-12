from adapenvs.farmworld.farmworld import Farmworld
from adapenvs.latent_env_wrapper import LatentWrapper, AgentLatentOptimizer

f = Farmworld({"num_agents": 1})

actions = dict([(i, 1) for i in range(1, f.num_agents+1)])
print("actions:", actions)

obs = f.step(actions)
print(obs)
f.reset()

fw = LatentWrapper(f)
obs = fw.reset()
print(obs)
obs = fw.step(actions)
print(obs)

obs =  fw.reset()
print(obs)