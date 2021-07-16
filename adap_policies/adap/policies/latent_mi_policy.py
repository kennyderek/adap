"""
PyTorch policy class used for PPO.
"""
import logging
from typing import Dict, List, Type, Union, Optional

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import (Postprocessing, compute_advantages)
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
from ray.rllib.evaluation.episode import MultiAgentEpisode

torch, nn = try_import_torch()

from torch.distributions.categorical import Categorical
from collections import OrderedDict

# my own imports
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import (DictFlatteningPreprocessor,
                                            get_preprocessor)

logger = logging.getLogger(__name__)

from .common import get_discrim_loss

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

    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # context_loss, baseline = get_mi_loss(policy, model, dist_class, train_batch, logits, curr_action_dist)
    
    mi_discrim_loss, baseline = get_discrim_loss(policy, model, dist_class, train_batch, logits, curr_action_dist)

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
            ) + mi_discrim_loss * policy.config["context_loss_coeff"]
        # total_loss = context_loss*policy.config["context_loss_coeff"]
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
    policy._mean_total_rew = torch.mean(train_batch[Postprocessing.VALUE_TARGETS]).item()

    policy._mi_discrim_loss = mi_discrim_loss.item()
    policy._baseline_discrim_loss = baseline
    # policy._rew_mod_mean = policy.model.get_rew_mod_and_mean()[0]
    # policy._rew_mod_sig = policy.model.get_rew_mod_and_mean()[1]

    return total_loss

from ray.rllib.agents.ppo.ppo_torch_policy import (
    EntropyCoeffSchedule, LearningRateSchedule, ValueNetworkMixin,
    apply_grad_clipping, kl_and_loss_stats,
    setup_config, setup_mixins)


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
        "mi_discrim_loss": policy._mi_discrim_loss,
        "baseline_discrim_loss": policy._baseline_discrim_loss,
        # "after_rew_mod_mean": policy._mean_total_rew,
        # "rew_mod_sig": policy._rew_mod_sig
    }

def vf_preds_fetches(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.

    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.

    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    # Return value function outputs. VF estimates will hence be added to the
    # SampleBatches produced by the sampler(s) to generate the train batches
    # going into the loss function.
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }

def compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.

    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """

    # print("batch shape:", sample_batch["obs"].shape)
    if policy.config["mi_mode"] in ["state", "state_action_non_diff"]:
        original_obs = restore_original_dimensions(
                torch.tensor(sample_batch["obs"]),
                policy.observation_space,
                tensorlib="torch"
            )

        # print("ORIGINAL OBS SHAPE:", original_obs['obs'].shape)

        if policy.config["mi_mode"] == "state":
            mi_obs = original_obs['obs']
        elif policy.config["mi_mode"] == "state_action_non_diff":
            # TODO, find a way to just reuse the action dist inputs from the original forward pass
            with torch.no_grad():
                action_dist_inputs, _ = policy.model.forward({"obs_flat": torch.tensor(sample_batch["obs"]), "obs": original_obs}, state=None, seq_lens=None)
            mi_obs = torch.cat([original_obs['obs'], torch.nn.functional.softmax(action_dist_inputs, dim=-1)], dim=-1)

        if policy.config["discrete_context"]:
            with torch.no_grad():
                ctx_pred = policy.model.predict_context(mi_obs)
            dist = Categorical(logits=ctx_pred)
            ctx_target = np.argmax(original_obs['ctx'].numpy(), axis=-1) # get the index of each one hot
            rew_mod = dist.log_prob(torch.tensor(ctx_target)).numpy() - np.log(1/policy.config['context_size'])
        else:
            with torch.no_grad():
                ctx_pred = policy.model.predict_context(mi_obs).data.numpy()
            ctx_target = original_obs['ctx'].data.numpy()
            
            # rew_mod = -np.mean((ctx_pred - ctx_target)**2, axis=-1)

            rew_mod = policy.model.get_rew_mod(ctx_pred, ctx_target)

        # rew_mod = (rew_mod - np.mean(rew_mod)) / (np.std(rew_mod) + 1e-8)
        # with torch.no_grad():
        #     action_dist_inputs, _ = policy.model.forward({"obs_flat": torch.tensor(sample_batch["obs"]), "obs": original_obs}, state=None, seq_lens=None)
        # if policy.config["clip_mi_rewards"] != 0:
        #     clip_val = policy.config["clip_mi_rewards"]
        #     sample_batch['rewards'] = sample_batch['rewards'] + np.clip(rew_mod, -clip_val, clip_val)*policy.config['mi_reward_scaling']
        # else:
        sample_batch['rewards'] = sample_batch['rewards']*policy.config['extrinsic_reward_scaling'] + rew_mod*policy.config['mi_reward_scaling']# + torch.distributions.Categorical(action_dist_inputs).entropy().numpy().flatten()

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = policy.model.get_input_dict(sample_batch, index="last")
        last_r = policy._value(**input_dict, seq_lens=input_dict.seq_lens)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True))

    return batch

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
MIPPOContextPolicy = build_policy_class(
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
        ValueNetworkMixin
    ],
)
