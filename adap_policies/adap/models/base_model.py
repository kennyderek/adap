import copy

import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.view_requirement import ViewRequirement
from gym.spaces import Box

th, nn = try_import_torch()

class CtxDiscrim(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        nn.Module.__init__(self)

        self.mi_decider = nn.Sequential(
            nn.Linear(input_size, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
        )
    
    def forward(self, obs):
        return self.mi_decider(obs)

class BaseModel(TorchModelV2, nn.Module):
    """PyTorch version of the CustomLossModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

        if model_config['custom_model_config']['value_nonlinearity'] == 'Tanh':
            self.value_nonlinearity = nn.Tanh
        elif model_config['custom_model_config']['value_nonlinearity'] == 'ReLU':
            self.value_nonlinearity = nn.ReLU
        else:
            assert False, "invalid value non-linearity!"

        self.clamp_mu_logits = model_config['custom_model_config']['clamp_mu_logits']

        self.hidden_dim = model_config['custom_model_config']['hidden_dim']
        self.context_size = model_config['custom_model_config']['context_size']
        self.use_mi_discrim = model_config['custom_model_config']['use_mi_discrim']
        if self.use_mi_discrim:
            self.mi_mode = model_config['custom_model_config']['mi_mode']
            assert self.mi_mode in ["state", "state_action_non_diff", "state_action_diff", "state_action_diff_noisy"]

        self.obs_space_size = obs_space.sample().shape[0]
        self.num_outputs = num_outputs

        self.agent_branch = None
        self.value_branch = None
        
        self._last_obs_seen = None

    def initialize_mi_discrim(self):
        if self.mi_mode == "state":
            print("REINITIALIZED")
            self.rew_mod_mean = nn.Parameter(th.tensor(0.0), requires_grad=True)
            self.rew_mod_sig = nn.Parameter(th.tensor(1.0), requires_grad=True)
            self.mi_decider = CtxDiscrim(self.get_input_size_excluding_ctx(), self.hidden_dim, self.context_size)
        elif "state_action" in self.mi_mode:
            self.rew_mod_mean = nn.Parameter(th.tensor(0.0), requires_grad=True)
            self.rew_mod_sig = nn.Parameter(th.tensor(1.0), requires_grad=True)
            self.mi_decider = CtxDiscrim(self.get_input_size_excluding_ctx() + self.num_outputs, self.hidden_dim, self.context_size)

    def get_input_size_inluding_ctx(self):
        assert False, "must override!"

    def get_input_size_excluding_ctx(self):
        assert False, "must override!"

    def predict_context(self, obs):
        return self.mi_decider(obs)

    def softsign_mu(self, logits):
        mu_logits, sig_logits = th.split(logits, int(self.num_outputs/2), dim=1)
        mu_logits = th.nn.functional.softsign(mu_logits)
        logits = th.cat([mu_logits, sig_logits], dim=-1)
        return logits

    def get_rew_mod(self, ctx_pred, ctx_target):
        rew_mod = -np.mean((ctx_pred - ctx_target)**2, axis=-1)
        mean = np.mean(rew_mod)
        sig = np.std(rew_mod)

        self.rew_mod_mean.data = self.rew_mod_mean.data*0.9 + mean*0.1
        self.rew_mod_sig.data = (0.9*self.rew_mod_sig.data**2 + 0.1*sig**2)**(1/2)

        # print(self.rew_mod_mean, self.rew_mod_sig)
        return (rew_mod - self.rew_mod_mean.data.item())#/ (self.rew_mod_sig.data.item() + 0.01)

    def get_rew_mod_and_mean(self):
        return self.rew_mod_mean.data.item(), self.rew_mod_sig.data.item()