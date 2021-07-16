from adap.latent_optimizer import evolve
import argparse
import yaml

from ray import tune

from common import get_env_and_callbacks, get_name_creator, get_trainer, build_trainer_config, get_name_creator

import copy

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--exp-name', type=str, default="context_exp")
parser.add_argument('--local-dir', type=str, default="~/ray_results")

parser.add_argument('--restore', type=str, default="") # path to restore the game
parser.add_argument('--evaluate', type=str, default="") # path to restore the game
parser.add_argument('--evolve', action="store_true") # path to restore the game

parser.add_argument("--train", action="store_true")
parser.add_argument("--conf", type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.conf.split("/")

    with open(args.conf, 'r') as f:
        config = yaml.load(f)
    print("This is the config\n", config)

    env_conf = config["env_conf"]
    training_conf = config["training_conf"]

    Env, MyCallbacks, simulator, set_latent_fn = get_env_and_callbacks(env_conf['env'])
    test_env = Env(env_conf)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    print("ACTION SPACE:", act_space)

    trainer_cls = get_trainer(training_conf['trainer'])
    trainer_conf = build_trainer_config(Env, MyCallbacks, env_conf, training_conf, obs_space, act_space)

    stop = {
        "timesteps_total": training_conf['timesteps_total'],
        "training_iteration": training_conf['training_iteration'],
        # "episode_reward_mean": 34 # this would mean 35/40 agents have survived on average, and is probably a good stop condition
    }

    if args.restore == "":
        tune.run(trainer_cls,
            config=trainer_conf,
            stop=stop,
            checkpoint_freq=25,
            checkpoint_at_end=True,
            local_dir=args.local_dir, # defaults to ~/ray_results
            name=args.exp_name, # the dir name after ~/ray_results
            trial_dirname_creator=get_name_creator(path), # the name after ~/ray_results/context_exp
        )
    elif args.train:
        # pick up where we left off training, using a checkpoint
        tune.run(trainer_cls,
            config=trainer_conf,
            stop=stop,
            checkpoint_freq=100,
            checkpoint_at_end=True,
            local_dir=args.local_dir,
            name=args.exp_name, # the dir name after ~/ray_results
            trial_dirname_creator=get_name_creator(path), # the name after ~/ray_results/context_exp
            restore=args.restore
        )
    else:
        import ray
        import random

        ray.init()
        agent = trainer_cls(trainer_conf)
        agent.restore(args.restore)

        if args.evaluate != "":
            with open(args.evaluate, 'r') as f:
                config = yaml.load(f)
            print("USING CONFIG:\n", config)
            env_conf = config["env_conf"]
        env_conf_ablate = copy.deepcopy(env_conf)

        use_latent = None
        if args.evolve:
            best_latents = evolve(lambda id: agent, Env, env_conf_ablate, set_latent_fn, evolve_steps=30)
            use_latent = best_latents[0]
        model_name = args.restore.split("/")[-3]
        
        for i in range(0, 5):
            simulator(lambda id: agent, Env, env_conf_ablate, ctx=use_latent, img_num=i, title=model_name)

