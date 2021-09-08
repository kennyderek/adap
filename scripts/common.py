from os import name

from adapenvs.gym_wrapper.gym_wrapper import GymWrapper

from adapenvs.simulators.simulation import simulate_field_with_context, step_farm_with_context, simulate_gym

from adap.policies.trainers import MyCallbacksFarm, MyCallbacksField, MyCallbacksSoccer
from ray.rllib.agents.callbacks import DefaultCallbacks

from adap.policies.adap_policy import PPOContextPolicy
from adap.policies.latent_mi_policy import MIPPOContextPolicy
from adap.policies.trainers import execution_plan_maker


from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo import (DEFAULT_CONFIG, execution_plan,
                                      get_policy_class, validate_config,
                                      warn_about_bad_reward_scales)

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.torch_policy_template import build_torch_policy

from datetime import datetime

from adap.latent_optimizer import set_latent_gym

from adap.models.concat_model import ConcatModel
from adap.models.mult_model import MultModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model(
    "ContextConcat", ConcatModel)
ModelCatalog.register_custom_model(
    "ContextMult", MultModel)


def get_name_creator(path):
    def name_creator(trial):
        name = path[-1].replace(".yaml", "") + \
            datetime.now().strftime("_%d_%H_%M_%S")
        return name
    return name_creator


def get_trainer(args_trainer):
    if args_trainer == "PPO":
        return PPOTrainer

    pi = None
    if args_trainer == "Context":
        pi = PPOContextPolicy
    elif args_trainer == "MIPPO":
        pi = MIPPOContextPolicy
    else:
        assert False, "improper trainer choice specified: choose one of PPO, Context, MIPPO"

    Trainer = build_trainer(
        name="Trainer",
        default_config=DEFAULT_CONFIG,
        validate_config=validate_config,
        default_policy=pi,
        # get_policy_class=get_policy_class,
        execution_plan=execution_plan_maker(standardize_adv=True),
    )
    return Trainer


def get_env_and_callbacks(env_name):
    if env_name == "Gym":
        return GymWrapper, DefaultCallbacks, simulate_gym, set_latent_gym
    else:
        assert False, "improper env choice"


def build_trainer_config(env_class,
                         callback_class,
                         env_conf,
                         trainer_conf,
                         obs_space,
                         act_space
                         ):
    if trainer_conf['model'] == "None":
        model_config = {}
    else:
        model_config = {
            "custom_model": trainer_conf['model'],
            "custom_model_config": {
                "context_size": env_conf["context_size"],
                "hidden_dim": trainer_conf['hidden_dim'],
                "use_mi_discrim": trainer_conf.get('use_mi_discrim', False),
                "mi_mode": trainer_conf.get("mi_mode", None),
                "lstm_state_size": trainer_conf.get('lstm_state_size', 0),
                "value_nonlinearity": trainer_conf.get('value_nonlinearity', 'Tanh'),
                "clamp_mu_logits": trainer_conf.get('clamp_mu_logits', False),
            }
        }

    policy_config = {
        "model": model_config,
        "count_steps_by": trainer_conf['count_steps_by'],  # or env_steps
        "rollout_fragment_length": trainer_conf['rollout_fragment_length'],

        "vf_loss_coeff": trainer_conf['vf_loss_coeff'],
        "entropy_coeff": trainer_conf['entropy_coeff'],

        "context_loss_coeff": trainer_conf.get('context_loss_coeff', 0),

        "context_size": env_conf["context_size"],
        "context_sampler": env_conf.get("context_sampler", "l2"),
        "num_state_samples": trainer_conf.get("num_state_samples", 30),  # 30
        # 10 usually
        "num_context_samples": trainer_conf.get("num_context_samples", 10),
        "context_competition_size": trainer_conf.get("context_competition_size", 0),
        "discrete_context": env_conf.get("discrete_context", False),
        "use_value_bias": trainer_conf.get("use_value_bias", False),

        "num_env_actions": env_conf['max_env_actions'],
        "action_dist": env_conf.get("action_dist", "categorical"),
        "bucket_threshold": 0,
        "context_epsilon": trainer_conf.get("context_epsilon", 0),

        "mi_reward_scaling": trainer_conf.get("mi_reward_scaling", 0),
        "extrinsic_reward_scaling": trainer_conf.get("extrinsic_reward_scaling", 1),

        "mi_mode": trainer_conf.get("mi_mode", None),
        "action_noise": trainer_conf.get("action_noise", 0),

        "cc_coeff_schedule": trainer_conf.get("cc_coeff_schedule", None),
        "entropy_coeff_schedule": trainer_conf.get("entropy_coeff_schedule", None),
    }

    config = {
        "env": env_class,
        "env_config": env_conf,

        "multiagent": {
            "policies": {
                # the first tuple value is None -> uses default policy
                "policy_1": (None, obs_space, act_space, policy_config),
            },
            "batch_mode": trainer_conf['batch_mode'],
            # "batch_mode": "truncate_episodes" if trainer == "ContextFlex" else "complete_episodes",
            "rollout_fragment_length": trainer_conf['rollout_fragment_length'],
            "count_steps_by": trainer_conf['count_steps_by'],  # or env_steps
            "policy_mapping_fn": lambda agent_id: "policy_1"
        },

        "rollout_fragment_length": trainer_conf['rollout_fragment_length'],

        "lambda": trainer_conf.get('lambda', 0.95),
        "gamma": trainer_conf.get('gamma', 0.99),  # discount factor on the MDP
        "clip_param": 0.2,
        "grad_clip": None if trainer_conf.get('grad_clip', 0.5) == "None" else trainer_conf.get('grad_clip', 0.5),
        "lr": trainer_conf['lr'],

        "train_batch_size": trainer_conf['train_batch_size'],
        "sgd_minibatch_size": trainer_conf['sgd_minibatch_size'],
        "num_sgd_iter": 10,

        "framework": "torch",
        "num_workers": 2,
        "num_gpus": 0,

        # "simple_optimizer": False,
        "callbacks": callback_class,
    }
    return config
