from adap.latent_optimizer import evolve
import argparse
import yaml

from common import get_env_and_callbacks, get_name_creator, get_trainer, build_trainer_config, get_name_creator

import copy

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--local-dir', type=str, default="~/ray_results")
parser.add_argument('--exp-name', type=str, default="context_exp", help="name of these experiments, will be a subdirectory of --local-dir")
parser.add_argument('--trial-name', type=str, default="", help="name of this specific seed, if empty will append the timestap to the .yaml name")

parser.add_argument('--env-conf', type=str, help="path to config file of train environment")
parser.add_argument("--train-conf", type=str, help="path to the config file containing ADAP hyperparameters and environment settings")
parser.add_argument("--model", type=str, default="ContextMult", help="model to use, in {mult, concat}")

parser.add_argument('--restore', type=str, default="", help="")
parser.add_argument('--evolve', action="store_true", help="whether to perform latent optimization")
parser.add_argument("--train", action="store_true", help="used to continue training a restored model")

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.train_conf.split("/")

    with open(args.train_conf, 'r') as f:
        training_conf = yaml.load(f)
        training_conf["model"] = args.model
    with open(args.env_conf, 'r') as f:
        env_conf = yaml.load(f)
    
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
    }

    if args.restore == "":
        from ray import tune

        tune.run(trainer_cls,
            config=trainer_conf,
            stop=stop,
            checkpoint_freq=25,
            checkpoint_at_end=True,
            local_dir=args.local_dir, # defaults to ~/ray_results
            name=args.exp_name, # the dir name after ~/ray_results
            trial_dirname_creator=get_name_creator(path, args.trial_name), # the name after ~/ray_results/context_exp
        )
    elif args.train:
        from ray import tune

        # pick up where we left off training, using a checkpoint
        tune.run(trainer_cls,
            config=trainer_conf,
            stop=stop,
            checkpoint_freq=100,
            checkpoint_at_end=True,
            local_dir=args.local_dir,
            name=args.exp_name, # the dir name after ~/ray_results
            trial_dirname_creator=get_name_creator(path, args.trial_name), # the name after ~/ray_results/context_exp
            restore=args.restore
        )
    else:
        import ray
        import random

        ray.init()
        agent = trainer_cls(trainer_conf)
        agent.restore(args.restore)

        use_latent = None
        if args.evolve:
            best_latents = evolve(lambda id: agent, Env, env_conf, set_latent_fn, evolve_steps=30)
            use_latent = best_latents[0]
        model_name = args.restore.split("/")[-3]
        
        for i in range(0, 5):
            simulator(lambda id: agent, Env, env_conf, ctx=use_latent, img_num=i, title=model_name)

