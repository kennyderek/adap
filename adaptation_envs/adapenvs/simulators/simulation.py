import matplotlib.pyplot as plt
from numpy.core.shape_base import stack
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
import numpy as np
import torch as th
from PIL import Image, ImageDraw, ImageFont
import random
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm
from copy import deepcopy
from adapenvs.utils.context_sampler import TRANSFORMS

def step_using_policy(agent_fn, obs, preprocessor):
    actions = {}
    with th.no_grad():
        for agent_id, obs in obs.items():
            actions[agent_id] = agent_fn(agent_id).get_policy("policy_1").compute_actions([preprocessor.transform(obs)])[0][0]
    return actions

def plot_mapped_contexts(context_size, model):
    context_inputs = th.rand(size=(1000, context_size))
    with th.no_grad():
        contexts = model.encode_contexts(context_inputs)
    plt.scatter(contexts.T[0], contexts.T[1], s=3)
    plt.savefig("contexts.png")

def upscale_rescale(matrix, scaling=None):
    # assumes input is [-1, 1]
    # matrix_rescaled = (matrix + 1) / 2
    matrix_rescaled = matrix

    # matrix_rescaled = th.tensor(matrix).float()
    if matrix_rescaled.shape[-1] < 3:
        right = th.tensor(np.array([[0, 1, 0],
                                    [1, 0, 0.2]])).float()
        matrix_rescaled = th.matmul(matrix_rescaled, right)
    if matrix_rescaled.shape[-1] > 3:
        # scaling = np.random.rand(matrix.shape[-1], 3)

        # scaling = scaling / np.sum(scaling, axis=0) # for continuous

        right = th.tensor(scaling).float()
        print(matrix_rescaled)
        matrix_rescaled = th.matmul(th.tensor(matrix_rescaled).float(), right).numpy()
    return matrix_rescaled[...,0:4]

def simulate_gym(agent, Env, env_conf, ctx=None, target="normal", render=True, title="gym_environment", stack_frames=True, img_num=0):
    if target == "noise":
        env_conf["mode"] = "noise"
        env_conf["noise_scale"] = 0.25

    env = Env(env_conf)
    obs = env.reset()
    score = 0

    preprocessor = DictFlatteningPreprocessor(env.observation_space)
    if isinstance(ctx, np.ndarray):
        # print("set context to: ", ctx)
        env.set_context(ctx)
        obs[0]['ctx'] = ctx

    imgs = []
    base = None
    stacks = 0
    for tstep in range(1000):
        if render:
            a = env.render(mode="rgb_array")
            if stack_frames:
                if tstep == 0:
                    base = a.astype(np.float)
                    stacks = 1
                elif tstep % 20 == 0:
                    base = np.minimum(a.astype(np.float), base)
                    stacks += 1
            else:
                imgs.append(Image.fromarray(a))
        
        actions = step_using_policy(agent, obs, preprocessor)
        obs, rew, done, i = env.step(actions)

        # if target in ["normal", "noise"]:
        #     score += rew[0]
        # elif target == "left":
        #     score -= obs[0]['obs'][0] # x axis pos
        # elif target == "right":
        #     score += obs[0]['obs'][0]
        # elif target == "sway":
        #     score += abs(obs[0]['obs'][3]) # pole angular velocity
        # elif target == "move_lr":
        #     score += abs(obs[0]['obs'][1]) # cart velocity
        # elif target == "still":
        #     score -= abs(obs[0]['obs'][1]) # cart velocity

        if done[0]:
            if render and stack_frames:
                imgs.append(Image.fromarray(base.astype(np.uint8)))
            break
    if render:
        if not stack_frames:
            imgs[0].save(fp="{}_{}.gif".format(title, img_num), format='GIF', append_images=imgs[1:],
                save_all=True, duration=20, loop=0)
        else:
            imgs[0].save(fp="{}_{}.png".format(title, img_num))

        env.close()
    return score

def simulate_field_with_context(agent, Env, env_conf, img_num = 0, img_root="", title=""):
    env = Env(env_conf)

    preprocessor = DictFlatteningPreprocessor(env.observation_space)

    for i in range(0, 5):
        obs = env.reset()
        paths = [[] for i in range(env_conf["num_agents"])]
        contexts = {}
        # Step the field environment in order to gather data
        while True:
            actions = step_using_policy(agent, obs, preprocessor)
            obs, rew, done, infos = env.step(actions)
            for agent_id in obs:
                # if agent_id != "__contexts__":
                contexts[agent_id] = obs[agent_id]['ctx']
                paths[agent_id].append(list(env.agent_positions[agent_id]))
            if done["__all__"]:
                break

        # Plot the context-dependent trajectories
        sl = env_conf['max_side']
        plt.axis([-sl, sl, -sl, sl])
        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FBBCF0']
        scaling = np.random.rand(env.context_size, 3)
        for i in range(0, env_conf["num_agents"]):
            p = np.array(paths[i]).T.astype(np.float)
            shift = (np.random.random() - 0.5) / 4
            p += shift
            plt.plot(p[0],p[1], color=upscale_rescale(contexts[i], scaling=scaling))
    
    # plot the goal locations
    plt.scatter([0], [0], marker="D", s=90, linewidths=5, c="gold", zorder=999)
    for goal in env.food_locations:
        plt.scatter([goal[0]], [goal[1]], marker="*", s=150, linewidths=5, c="gold", zorder=1000)
    
    plt.savefig("{}{}field_trajectories_{}.png".format(img_root, title, img_num))
    plt.clf()

def farm_reset_func(env):
    obs = env.reset(pixel_viz=True)
    return obs


def simulate_soccer(agentA,
                    agentB,
                    Env,
                    env_conf,
                    policy_mapping_fn=None,
                    contexts_for_A=None,
                    contexts_for_B=None,
                    steps=1000,
                    mode="normal",
                    trainer=True):
    env = Env(env_conf)
    obs = env.reset()
    preprocessor = DictFlatteningPreprocessor(env.observation_space)

    A = 0
    B = 0
    Draw = 0
    winning = []
    losing = []
    draw = []
    for i in range(0, steps):
        obs = env.reset()

        if contexts_for_A != None:
            env.set_A_context(random.choice(contexts_for_A))
        if contexts_for_B != None:
            env.set_B_context(random.choice(contexts_for_B))

        # env.set_B_context(np.array([-0.76925718,  0.13451791,  0.62461855]))
        while True:
            actions = step_using_policy(lambda id: agentA if id == 0 else agentB, obs, preprocessor)
            obs, rew, done, infos = env.step(actions)
            if done["__all__"]:
                if rew[0] == 1:
                    A += 1
                    winning.append(env.get_B_context())
                elif rew[1] == 1:
                    B += 1
                    losing.append(env.get_B_context())
                else:
                    Draw += 1
                    draw.append(env.get_B_context())
                break

    # if policy_mapping_fn != None:
    #     print("Winners:\n Adversarial: {} Predefined: {} Draw: {}".format(A, B, Draw))
    # else:
    #     print("Winners:\n A: {} B: {} Draw: {}".format(A, B, Draw))

    return A, B, Draw

def simulate_soccer_img(agentA,
                        agentB,
                        Env,
                        env_conf,
                        img_folder="",
                        img_num = 0,
                        contexts_for_A=None,
                        contexts_for_B=None,
                        policy_mapping_fn=None,
                        mode="normal"):
    print("using latents for B:", contexts_for_B)
    env = Env(env_conf)
    obs = env.reset()
    preprocessor = DictFlatteningPreprocessor(env.observation_space)

    # if isinstance(set_context, np.ndarray):
    #     env.set_A_context(set_context)
    if contexts_for_A != None:
        env.set_A_context(random.choice(contexts_for_A))
    if contexts_for_B != None:
        b = random.choice(contexts_for_B)
        env.set_B_context(b)
        obs[1]['ctx'] = b
    
    size = 500
    imgs = []
    scale_y = int(size * 5 / 5)
    scale_x = int(size * 4 / 5)
    imgs.append(Image.fromarray(env.get_game_board() * 255).resize((scale_y, scale_x), Image.NEAREST))

    while True:
        actions = step_using_policy(lambda id: agentA if id == 0 else agentB, obs, preprocessor)
        # actions = step_using_policy_soccer(agentA, agentB, obs, preprocessor, policy_mapping_fn)
        obs, rew, done, infos = env.step(actions)

        img = Image.fromarray(env.get_game_board() * 255).resize((scale_y, scale_x), Image.NEAREST).convert('RGB')
        if done["__all__"]:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("utils/arial.ttf", 120)
            # font = ImageFont.truetype("sans-serif.ttf", 16)
            if rew[1] == -1:
                draw.text((4, 4),"A Wins!", (0,255,0), font=font)
            elif rew[1] == 1:
                draw.text((4, 4),"B Wins!", (255,0,0), font=font)
            else:
                draw.text((4, 4),"Draw!", (0,0,255), font=font)
            imgs.append(img)
            break
        imgs.append(img)

    
    imgs[0].save(fp="{path}{title}_playback_{num}.gif".format(path=img_folder, title="markov_soccer", num=img_num), format='GIF', append_images=imgs[1:],
        save_all=True, duration=200, loop=0)


def step_farm_with_context(agent_fn, Env, env_conf, img_num = 0, save_img=True, farm_reset_func=farm_reset_func, title="farm_test"):
    env = Env(env_conf)
    preprocessor = DictFlatteningPreprocessor(env.observation_space)

    env.reset(pixel_viz=save_img)

    obs = farm_reset_func(env)

    size = 400
    imgs = []
    scale_y = int(size * env.sidelength_x / max(env.sidelength_x, env.sidelength_y))
    scale_x = int(size * env.sidelength_y / max(env.sidelength_x, env.sidelength_y))
    imgs.append(Image.fromarray(env.get_world_map()).resize((scale_y, scale_x), Image.NEAREST))
    for i in range(0, env_conf["eps_length"]+1):
        actions = step_using_policy(agent_fn, obs, preprocessor)
        obs, rew, done, infos = env.step(actions)

        if save_img:
            # scale_x = int(250 * env.sidelength_x / max(env.sidelength_x, env.sidelength_y))
            # scale_y = int(250 * env.sidelength_y / max(env.sidelength_x, env.sidelength_y))
            imgs.append(Image.fromarray(env.get_world_map()).resize((scale_y, scale_x), Image.NEAREST))
        if done["__all__"]:
            break

    if save_img:
        imgs[0].save(fp="{title}_playback_{num}.gif".format(title=title, num=img_num), format='GIF', append_images=imgs[1:],
            save_all=True, duration=200, loop=0)


def on_agent_done(env, agent):
    ctx = agent.get_context()
    avc = agent.chicken_attacks
    avt = agent.tower_attacks
    return ctx, avc, avt
