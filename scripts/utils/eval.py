from adapenvs.farmworld.agent import Agent
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
import torch as th
import ray
import yaml
from ray.rllib.agents.callbacks import DefaultCallbacks
from common import get_trainer, build_trainer_config
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from adapenvs.latent_env_wrapper import AgentLatentOptimizer

def step_using_policy(agent_fn, obs, preprocessor):
    actions = {}
    with th.no_grad():
        for agent_id, obs in obs.items():
            actions[agent_id] = agent_fn(agent_id).get_policy("policy_1").compute_actions([preprocessor.transform(obs)])[0][0]
    return actions

def playthrough_simple(agent_fn, env, env_conf, save_img=False, title="", img_num=1, max_size = 150, speed = 20):
    # speed should be 200 for markov soccer and farmworld

    preprocessor = DictFlatteningPreprocessor(env.observation_space)

    obs = env.reset()

    frames = None
    scale_x = None
    scale_y = None
    if save_img:
        frames = []
        first_frame = env.render()
        shape = first_frame.shape # assumes (w, h) or (w, h, c)
        size = max_size
        side_x = shape[1]
        side_y = shape[0]
        scale_y = int(size * side_y / max(side_x, side_y))
        scale_x = int(size * side_x / max(side_x, side_y))
        frames.append(Image.fromarray(first_frame).resize((scale_x, scale_y), Image.NEAREST))

    game_data = dict((agent_id, defaultdict(list)) for agent_id in range(1, env.num_agents+1))
    for iter in range(0, env_conf["eps_length"]+1):
        actions = step_using_policy(agent_fn, obs, preprocessor)
        obs, rew, done, infos = env.step(actions)

        for agent_id in range(1, env.num_agents+1):
            game_data[agent_id]['obs'] += [obs.get(agent_id)] if obs.get(agent_id) is not None else []
            game_data[agent_id]['rew'] += [rew.get(agent_id)] if rew.get(agent_id) is not None else []
            game_data[agent_id]['done'] += [done.get(agent_id)] if done.get(agent_id) is not None else []
            game_data[agent_id]['infos'] += [infos.get(agent_id)] if infos.get(agent_id) is not None else []

        if save_img:
            frames.append(Image.fromarray(env.render()).resize((scale_x, scale_y), Image.NEAREST))

        if done["__all__"]:
            if save_img:
                # pad to an appropriate GIF length, for GIF consistency across methods
                (w, h) = frames[-1].size
                empty = Image.fromarray(np.zeros((h, w)))
                while len(frames) < env_conf['eps_length']:
                    frames.append(empty)
            break

    if save_img:
        frames[0].save(fp="{title}_playback_{num}.gif".format(title=title, num=img_num), format='GIF', append_images=frames[1:],
            save_all=True, duration=speed, loop=0)

    return game_data

def try_adaptation(agent_fn, env, env_conf, steps = 150):
    opts = {}
    for agent_id in range(1, env.num_agents+1):
        opt = AgentLatentOptimizer(env.latent_size, env.default_latent_sampler, steps)
        opts[agent_id] = opt
        env.set_sampler_fn(agent_id, opt.sampler_fn)
    
    for _ in range(steps):
        env.reset()
        game_data = playthrough_simple(agent_fn, env, env_conf)

        for agent_id in range(1, env.num_agents+1):
            score = sum(game_data[agent_id]['rew'])

            opts[agent_id].update(score)

    return dict((agent_id, opt.get_best_fn()) for agent_id, opt in opts.items())


def get_mean_agent_reward(game_data):
    rews = []
    for agent_id, agent_dict in game_data.items():
        rews.append(sum(agent_dict['rew']))
    return np.mean(rews)

def get_mean_info_metric(game_data, info_metric):
    info = []
    for agent_id, agent_dict in game_data.items():
        info_for_agent = []
        for info_at_step in agent_dict['infos']:
            if info_metric in info_at_step:
                info_for_agent.append(info_at_step[info_metric])

        info.append(sum(info_for_agent))
    return np.mean(info)

def evaluate_on_ablation(agent,
                         Env,
                         ablation_env_conf_path,
                         result_name,
                         infos = [],
                         save_imgs = True,
                         eval_steps = 15,
                         gif_speed=200,
                         img_size=200):
    with open(ablation_env_conf_path, 'r') as f:
        env_conf = yaml.load(f)
    best = try_adaptation(lambda a: agent, Env(env_conf), env_conf, steps=eval_steps)
    env = Env(env_conf)
    for agent_id, fn in best.items():
        env.set_sampler_fn(agent_id, fn)

    if save_imgs:
        # save images
        for i in range(3):
            playthrough_simple(lambda a: agent,
                            env,
                            env_conf,
                            save_img=True,
                            title=result_name,
                            img_num=i,
                            speed=gif_speed,
                            max_size=img_size)

    # run evaluation metrics
    metrics = dict((metric, []) for metric in infos + ["reward"])
    for i in range(0, 30):
        game_data = playthrough_simple(lambda a: agent, env, env_conf, save_img=False)
        
        metrics['reward'].append(get_mean_agent_reward(game_data))
        for info_metric in infos:
            metrics[info_metric].append(get_mean_info_metric(game_data, info_metric))
    final_metrics = {}
    for metric, scores in metrics.items():
        final_metrics[metric] = np.mean(scores)
    return final_metrics

