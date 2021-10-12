import argparse
import yaml
from utils.eval import try_adaptation, playthrough_simple, evaluate_on_ablation
import json
from common import get_env_and_callbacks, get_name_creator, get_trainer, build_trainer_config, get_name_creator

import os
from adapenvs.farmworld.farmworld import Farmworld
from adapenvs.latent_env_wrapper import LatentWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from adap.policies.trainers import MyCallbacksFarm

class FarmworldWrapped(LatentWrapper, MultiAgentEnv):

    def __init__(self, config):
        env = Farmworld(config)
        super().__init__(env)

BASE_ABLATION_PATH = "../configs/farmworld/ablations/"
ABLATIONS_PATHS = [BASE_ABLATION_PATH + "far_corner.yaml",
                BASE_ABLATION_PATH + "speed.yaml",
                BASE_ABLATION_PATH + "patience.yaml",
                BASE_ABLATION_PATH + "poison_chickens.yaml",
                BASE_ABLATION_PATH + "wall_barrier.yaml"]

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--local-dir', type=str, default="~/ray_results")
parser.add_argument('--exp-name', type=str, default="context_exp", help="name of these experiments, will be a subdirectory of --local-dir")
parser.add_argument('--trial-name', type=str, default="", help="name of this specific seed, if empty will append the timestap to the .yaml name")

parser.add_argument('--env-conf', type=str, help="path to config file of train environment")
parser.add_argument("--train-conf", type=str, help="path to the config file containing ADAP hyperparameters and environment settings")
parser.add_argument("--model", type=str, default="mult", help="model to use, in {mult, concat}")

parser.add_argument('--restore', type=str, default="", help="")

parser.add_argument('--test-mode', action="store_true", help="run a shortened versioon of this, to test for bugs")

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.train_conf.split("/")

    with open(args.train_conf, 'r') as f:
        training_conf = yaml.load(f)
        training_conf["model"] = args.model
    with open(args.env_conf, 'r') as f:
        env_conf = yaml.load(f)

    Env = FarmworldWrapped
    callbacks = MyCallbacksFarm
    test_env = Env(env_conf)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    trainer_cls = get_trainer(training_conf['trainer'])
    trainer_conf = build_trainer_config(Env, callbacks, env_conf, training_conf, obs_space, act_space)

    stop = {
        "timesteps_total": training_conf['timesteps_total'] if not args.test_mode else 100,
        "training_iteration": training_conf['training_iteration'],
    }

    if args.restore == "":
        from ray import tune

        exp = tune.run(trainer_cls,
            config=trainer_conf,
            stop=stop,
            checkpoint_freq=1000,
            checkpoint_at_end=True,
            local_dir=args.local_dir, # defaults to ~/ray_results
            name=args.exp_name, # the dir name after ~/ray_results
            trial_dirname_creator=get_name_creator(path, args.trial_name), # the name after ~/ray_results/context_exp
        )
        last_checkpoint = exp.get_last_checkpoint()
    else:
        last_checkpoint = args.restore
        import ray
        ray.init()
    
    # e.g. We want adap_10_19_36_03 from .../ray_results/context_exp/adap_10_19_36_03/checkpoint_000001/checkpoint-1
    trial_name = last_checkpoint.split("/")[-3]

    agent = trainer_cls(trainer_conf)
    agent.restore(last_checkpoint)

    ablation_results = {}
    if not os.path.isdir("trials/"):
        os.mkdir("trials/")
    os.mkdir("trials/{}".format(trial_name))

    for ablation_path in ABLATIONS_PATHS:
        title = ablation_path.split("/")[-1].replace(".yaml", "")
        ablation_eval_name = "trials/{}/{}_{}".format(trial_name, trial_name, title)
        eval_metrics = evaluate_on_ablation(agent,
                                            Env,
                                            ablation_path,
                                            ablation_eval_name,
                                            eval_steps=30 if args.test_mode else 200,
                                            img_size=200,
                                            infos=["avc", "avt", "c_t_attacktropy"])
        ablation_results[title] = eval_metrics

    title = "train_no_evo"
    ablation_eval_name = "trials/{}/{}_{}".format(trial_name, trial_name, title)
    eval_metrics = evaluate_on_ablation(agent,
                                        Env,
                                        args.env_conf,
                                        ablation_eval_name,
                                        eval_steps=0,
                                        img_size=200,
                                        infos=["avc", "avt", "c_t_attacktropy"])
    ablation_results[title] = eval_metrics

    title = "train_with_evo"
    ablation_eval_name = "trials/{}/{}_{}".format(trial_name, trial_name, title)
    eval_metrics = evaluate_on_ablation(agent,
                                        Env,
                                        args.env_conf,
                                        ablation_eval_name,
                                        eval_steps=30 if args.test_mode else 200,
                                        img_size=200,
                                        infos=["avc", "avt", "c_t_attacktropy"])
    ablation_results[title] = eval_metrics

    # record ablation metrics to file
    with open("trials/{}/{}_metrics.json".format(trial_name, trial_name), 'w+') as f:
        f.write(json.dumps(ablation_results))