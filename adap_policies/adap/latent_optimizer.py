from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from adapenvs.utils.context_sampler import TRANSFORMS, SAMPLERS
import random
import numpy as np

from adapenvs.simulators.simulation import step_using_policy


def get_latent_score(agent_fn, env, set_latent_fn, sampler_fn, num_evals, **kwargs):

    scores = None
    latent_score_pairs = []
    obs_preprocessor = DictFlatteningPreprocessor(env.observation_space)

    env, obs, latents = set_latent_fn(env, sampler_fn, *kwargs)
    scores = {agent_id: 0 for agent_id in obs.keys()}

    for _ in range(0, num_evals):

        env.reset()
        env, obs, _ = set_latent_fn(env, random_choice_sampler(latents), *kwargs)

        while True:
            actions = step_using_policy(agent_fn, obs, obs_preprocessor)
            obs, rews, dones, infos = env.step(actions)

            for agent_id, rew in rews.items():
                scores[agent_id] += rew
            
            if dones['__all__']:
                break
    
    latent_score_pairs = [(latents[i], scores[i]) for i in latents.keys()]
    return latent_score_pairs

def set_latent_farmworld(env, sampler_fn):
    latents = {}
    for id, agent in env.agents.items():
        z = sampler_fn(id)
        # print("SAMPLED LATENT:", z)
        latents[id] = z
        agent.context = z
    obs = env._obs()
    return env, obs, latents

def set_latent_soccer(env, sampler_fn, contexts_for_A=None):
    b_id = 1
    z = sampler_fn(b_id)
    env.set_B_context(z)
    if contexts_for_A != None:
        env.set_A_context(random.choice(contexts_for_A))
    obs = env._obs()
    return env, obs, {1: z}

def set_latent_gym(env, sampler_fn):
    z = sampler_fn(id)
    obs = env.reset(latent_z = z)
    return env, obs, {0: z}

def random_choice_sampler(best):
    def sample(id):
        return random.choice(best)
    return sample

def assigned_sampler(latents):
    def sample(id):
        return latents[id]
    return sample

def evolve(
    agent_fn,
    Env,
    env_conf,
    set_latent_fn,
    get_latent_score_fn=get_latent_score,
    evolve_steps=150,
    random_thres=0.6,
    num_latent_evals=1,
    replication_thres=0.3,
    mutation_thres=0.75,
    finetuning_ratio=3/4,
    num_keep=4,
    **kwargs
):
    best = []

    def get_sampler_fn(estep):
        def sample(id):
            if (random.random() < random_thres or len(best) < 10) and estep < int(evolve_steps * finetuning_ratio):
                z = SAMPLERS['l2'](ctx_size=env_conf['context_size'], num=1)[0]
            else:
                r = random.random()
                if r < replication_thres:
                    z = random.choice(best[0:10])[0]
                elif r < mutation_thres and estep < int(evolve_steps * finetuning_ratio):
                    choice = random.choice(best[0:10])[0]
                    z = choice + np.random.random(choice.shape) / 20
                    z = TRANSFORMS['l2'](z)
                else:
                    z = best.pop(random.randint(0, num_keep))[0]
            return z
        return sample

    for estep in range(0, evolve_steps):
        latent_score_pairs = get_latent_score_fn(agent_fn, Env(env_conf), set_latent_fn, get_sampler_fn(estep), num_latent_evals, *kwargs)
        print("Latent vector {} has score {}".format(*latent_score_pairs[0]))
        best.extend(latent_score_pairs)
        best.sort(key=lambda t: t[1], reverse=True)

    return [c[0] for c in best[:num_keep]]
