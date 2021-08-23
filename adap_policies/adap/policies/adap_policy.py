"""
PyTorch policy class used for PPO.
"""
import logging
from typing import Dict, List, Type, Union, Optional

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import (Postprocessing,
                                                 compute_gae_for_sample_batch, compute_advantages)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import (EntropyCoeffSchedule,
                                           LearningRateSchedule)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import (apply_grad_clipping,
                                       convert_to_torch_tensor,
                                       explained_variance, sequence_mask)
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.evaluation.episode import MultiAgentEpisode

torch, nn = try_import_torch()

from collections import OrderedDict

# my own imports
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import (DictFlatteningPreprocessor,
                                            get_preprocessor)

logger = logging.getLogger(__name__)

from .common import get_context_kl_loss


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    # if policy.config["discrete_context"]:
    #     if policy.config["use_value_bias"]:
    #         context_loss = get_context_kl_loss_discrete_onehot_value(policy, model, dist_class, train_batch)
    #     else:
    #         context_loss = get_context_kl_loss_discrete_onehot(policy, model, dist_class, train_batch)
    # else:
    #     if policy.config["use_value_bias"]:
    #         context_loss = get_context_kl_loss_value(policy, model, dist_class, train_batch)
    #     else:
    context_loss = get_context_kl_loss(policy, model, dist_class, train_batch)

    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        assert False, "we shouldn't have state right now"
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    if policy.config["use_gae"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
        total_loss = reduce_mean_valid(
                -surrogate_loss + \
                policy.config["vf_loss_coeff"] * vf_loss + \
                -policy.entropy_coeff * curr_entropy
            ) + context_loss*policy.config["context_loss_coeff"]
        # total_loss = reduce_mean_valid(
        #     -surrogate_loss + #policy.kl_coeff * action_kl +
        #     policy.config["vf_loss_coeff"] * vf_loss -
        #     policy.entropy_coeff * curr_entropy) + context_loss
    else:
        # mean_vf_loss = 0.0
        # total_loss = reduce_mean_valid(-surrogate_loss +
        #                                #policy.kl_coeff * action_kl -
        #                                policy.entropy_coeff * curr_entropy) + context_loss
        pass

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl
    policy._context_loss = context_loss.item()

    return total_loss

from ray.rllib.agents.ppo.ppo_torch_policy import (
    EntropyCoeffSchedule, LearningRateSchedule, ValueNetworkMixin,
    apply_grad_clipping, compute_gae_for_sample_batch, kl_and_loss_stats,
    setup_config, setup_mixins, vf_preds_fetches)


class KLCoeffMixin:
    """Assigns the `update_kl()` method to the PPOPolicy.

    This is used in PPO's execution plan (see ppo.py) for updating the KL
    coefficient after each learning step based on `config.kl_target` and
    the measured KL value (from the train_batch).
    """

    def __init__(self, config):
        # The current KL value (as python float).
        self.kl_coeff = config["kl_coeff"]
        # Constant target value.
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        # if sampled_kl > 2.0 * self.kl_target:
        #     self.kl_coeff *= 1.5
        # elif sampled_kl < 0.5 * self.kl_target:
        #     self.kl_coeff *= 0.5
        # Return the current KL value.
        return self.kl_coeff

def loss_stats(policy: Policy,
                      train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for PPO. Returns a dict with important KL and loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy._total_loss,
        "policy_loss": policy._mean_policy_loss,
        "vf_loss": policy._mean_vf_loss,
        "vf_explained_var": policy._vf_explained_var,
        "kl": policy._mean_kl,
        "entropy": policy._mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
        "context_loss": policy._context_loss,
        "context_loss_coeff": policy.config['context_loss_coeff']
    }

class ContextCoeffSchedule:
    """Mixin for TorchPolicy that adds context coeff increase??"""

    def __init__(self, coeff, schedule):
        # self.entropy_coeff = entropy_coeff
        # self.context_running_mean = []
        # self.lowest_context_loss = 1 # value, update step
        # self.updates_since_lowest = 0
        # self.num_context_dimensions = 1

        if schedule is None:
            self.context_coeff_schedule = ConstantSchedule(
                coeff, framework=None)
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(schedule, list):
                self.context_coeff_schedule = PiecewiseSchedule(
                    schedule,
                    outside_value=schedule[-1][-1],
                    framework=None)
            else: # it would be an integer then
                # Implements previous version but enforces outside_value
                self.context_coeff_schedule = PiecewiseSchedule(
                    [[0, coeff], [coeff, 0.0]],
                    outside_value=0.0,
                    framework=None)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        self.config['context_loss_coeff'] = self.context_coeff_schedule.value(
            global_vars["timestep"])
        # self.config['context_loss_coeff'] = min(global_vars["timestep"] / 10_000, 0.5)
        # self.config['context_loss_ coeff']

def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ContextCoeffSchedule.__init__(policy, config["context_loss_coeff"], config["cc_coeff_schedule"])

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
PPOContextPolicy = build_policy_class(
    name="PPOContextPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, ContextCoeffSchedule
    ],
)
