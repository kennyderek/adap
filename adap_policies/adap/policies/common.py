
from math import dist
from typing import Dict, List, Type, Union

import numpy as np
import ray
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (TorchCategorical,
                                                      TorchDiagGaussian,
                                                      TorchDistributionWrapper)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from collections import OrderedDict, defaultdict

from ray.rllib.models.modelv2 import restore_original_dimensions
from utils.context_sampler import SAMPLERS, TRANSFORMS

# in this code, we call the latent z a 'context'. The code will be updated to reflect the latest terminology.
def get_context_kl_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch):
    if policy.config['action_dist'] == 'gaussian':
        logit_size = policy.config['num_env_actions']*2
        def transform_logits(logits):
            (mus, log_std) = torch.split(logits, policy.config['num_env_actions'], dim=1)
            stds = torch.exp(log_std) + 1/20
            return torch.distributions.Normal(mus, stds)
    elif policy.config['action_dist'] == 'categorical':
        logit_size = policy.config['num_env_actions']
        def transform_logits(logits):
            probs = nn.functional.softmax(logits, dim=-1) + torch.ones(logits.shape) / 20
            probs = probs / torch.sum(probs, dim=-1).unsqueeze(1)
            return torch.distributions.Categorical(probs=probs)
        
    original_obs = restore_original_dimensions(
            train_batch["obs"],
            policy.observation_space,
            tensorlib="torch"
        )
    
    context_size = policy.config['context_size'] # 4, 16, 64
    num_context_samples = policy.config['num_context_samples'] # 8
    num_state_samples = policy.config['num_state_samples'] # 50

    indices = torch.randperm(original_obs['obs'].shape[0])[:num_state_samples]
    sampled_states = original_obs['obs'][indices]
    num_state_samples = min(num_state_samples, sampled_states.shape[0])

    all_contexts = []
    all_action_dists = []
    for i in range(0, num_context_samples): # 10 sampled contexts
        sampled_context = SAMPLERS[policy.config['context_sampler']](ctx_size=context_size, num=1, torch=True)

        all_contexts.append(sampled_context)
        sampled_context = torch.cat([sampled_context]*num_state_samples, dim=0) # expect shape (20, 16)
        context_train_batch = torch.cat([sampled_context, sampled_states], dim=-1)
        obs_original_dim = OrderedDict([('ctx', sampled_context), ('obs', sampled_states)])
        logits, state = model.forward({"obs_flat": context_train_batch, "obs": obs_original_dim}, state=None, seq_lens=None)
        assert logits.shape == (num_state_samples, logit_size), logits.shape

        context_action_dist = transform_logits(logits)
        all_action_dists.append(context_action_dist)
    
    count = 0
    all_CLs = []
    all_ents = []
    for i in range(0, num_context_samples - 1):
        for j in range(i+1, num_context_samples):
            context_dist = torch.sum(torch.abs(all_contexts[i] - all_contexts[j])) / context_size
            if context_dist > 0:
                count += 1
                assert 0 < context_dist <= 2, context_dist #TODO change to 0<
                dist_1 = all_action_dists[i]
                dist_2 = all_action_dists[j]
                
                ent = dist_1.entropy() + dist_2.entropy()
                div = torch.mean(torch.exp(-torch.distributions.kl.kl_divergence(dist_1, dist_2)))
                assert torch.sum(torch.isnan(div)) == 0, "found nan value: " + str(div)
                all_CLs.append(div)
                all_ents.append(ent)
    
    return sum(all_CLs)/len(all_CLs)



def get_discrim_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper], train_batch: SampleBatch, action_logits, curr_action_dist):
    original_obs = restore_original_dimensions(
            train_batch["obs"],
            policy.observation_space,
            tensorlib="torch"
        )

    if policy.config['action_dist'] == 'gaussian':
        def transform_logits(logits):
            # print(torch.split(logits, policy.config['num_env_actions'], dim=1))
            (mus, log_std) = torch.split(logits, policy.config['num_env_actions'], dim=1)
            stds = torch.exp(log_std)
            return torch.cat([mus, log_std], dim=-1)
    elif policy.config['action_dist'] == 'categorical':
        def transform_logits(logits):
            return torch.nn.functional.softmax(logits, dim=-1)

    policy_action_noise = policy.config['action_noise']

    if policy.config['mi_mode'] == "state":
        mi_obs = original_obs['obs']
    elif policy.config['mi_mode'] == "state_action_diff":
        mi_obs = torch.cat([original_obs['obs'], transform_logits(action_logits)], dim=-1)
    elif policy.config['mi_mode'] == "state_action_diff_noisy":
        action_probs = transform_logits(action_logits)
        noise = torch.rand(action_probs.shape) * policy_action_noise
        action_probs = (action_probs + noise) / torch.sum(action_probs + noise, dim=-1).unsqueeze(1)
        mi_obs = torch.cat([original_obs['obs'], action_probs + noise], dim=-1)
    elif policy.config['mi_mode'] == "state_action_non_diff":
        mi_obs = torch.cat([original_obs['obs'], transform_logits(action_logits.detach().clone())], dim=-1)
    else:
        assert False, "wrong mi mode!"

    if policy.config['discrete_context']:
        loss_fn = nn.CrossEntropyLoss()
        ctx_pred = model.predict_context(mi_obs)
        targets = torch.argmax(original_obs['ctx'], dim=-1)
        loss = loss_fn(ctx_pred, targets)
        baseline = -100
    else:
        ctx_pred = model.predict_context(mi_obs)
        loss = torch.mean((ctx_pred - original_obs['ctx'])**2)
        baseline = torch.mean((torch.zeros(original_obs['ctx'].shape) - original_obs['ctx'])**2)

    return loss, baseline
