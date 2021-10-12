import argparse
import yaml
from utils.eval import try_adaptation, playthrough_simple, evaluate_on_ablation
import json
from common import get_env_and_callbacks, get_name_creator, get_trainer, build_trainer_config, get_name_creator

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--local-dir', type=str, default="~/ray_results")
parser.add_argument('--exp-name', type=str, default="context_exp", help="name of these experiments, will be a subdirectory of --local-dir")
parser.add_argument('--trial-name', type=str, default="", help="name of this specific seed, if empty will append the timestap to the .yaml name")

parser.add_argument('--env-conf', type=str, help="path to config file of train environment")
parser.add_argument("--train-conf", type=str, help="path to the config file containing ADAP hyperparameters and environment settings")
parser.add_argument("--model", type=str, default="mult", help="model to use, in {mult, concat}")

parser.add_argument('--restore', type=str, default="", help="")
parser.add_argument('--evolve', action="store_true", help="whether to perform latent optimization")

parser.add_argument('--evolve', action="store_true", help="whether to perform latent optimization")

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.train_conf.split("/")

    with open(args.train_conf, 'r') as f:
        training_conf = yaml.load(f)
        training_conf["model"] = args.model
    with open(args.env_conf, 'r') as f:
        env_conf = yaml.load(f)

    Env, callbacks = get_env_and_callbacks(env_conf['env'])
    test_env = Env(env_conf)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    trainer_cls = get_trainer(training_conf['trainer'])
    trainer_conf = build_trainer_config(Env, callbacks, env_conf, training_conf, obs_space, act_space)

    stop = {
        "timesteps_total": training_conf['timesteps_total'],
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

    ablation_eval_name = "{}_move_left".format(trial_name)
    eval_metrics = evaluate_on_ablation(agent,
                         Env,
                         "../configs/cartpole/ablations/move_left.yaml",
                         ablation_eval_name,
                         eval_steps=15)
    
    # record ablation metrics to file
    with open("{}_metrics.json".format(ablation_eval_name), 'w+') as f:
        f.write(json.dumps(eval_metrics))