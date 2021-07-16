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

from adap.models.base_model import BaseModel

th, nn = try_import_torch()

class MultModel(BaseModel):
    """TODO"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

        self.context_size = model_config['custom_model_config']['context_size']
        
        self.agent_branch_1 = nn.Sequential(
            nn.Linear(self.get_input_size_excluding_ctx(), self.hidden_dim),
            nn.Tanh()
        )
        self.agent_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.context_size),
            nn.Tanh()
        )
        self.agent_branch_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, num_outputs)
        )

        self.value_branch_1 = nn.Sequential(
            nn.Linear(self.get_input_size_excluding_ctx(), self.hidden_dim),
            self.value_nonlinearity()
        )
        self.value_scaling = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.context_size),
            self.value_nonlinearity()
        )
        self.value_branch_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.value_nonlinearity(),
            nn.Linear(self.hidden_dim, 1)
        )

        if self.use_mi_discrim:
            self.initialize_mi_discrim()

        self._last_obs_seen = None
    
    def get_input_size_excluding_ctx(self):
        return self.obs_space_size - self.context_size

    def get_input_size_inluding_ctx(self):
        return self.obs_space_size
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        observations = input_dict["obs"]['obs'].float()
        contexts = input_dict["obs"]['ctx'].float()
        self._last_obs_seen = (observations, contexts)

        batch_size = observations.shape[0]
        x = self.agent_branch_1(observations)
        x_a = self.agent_scaling(x)
        x_a = x_a.view((batch_size, self.hidden_dim, self.context_size)) # reshape to do context multiplication
        x_a_out = th.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        logits = self.agent_branch_2(x + x_a_out)

        if self.clamp_mu_logits:
            logits = self.softsign_mu(logits)

        return logits, state

    @override(ModelV2)
    def value_function(self):
        observations, contexts = self._last_obs_seen

        batch_size = observations.shape[0]
        x = self.value_branch_1(observations)
        x_a = self.value_scaling(x)
        x_a = x_a.view((batch_size, self.hidden_dim, self.context_size)) # reshape to do context multiplication
        x_a_out = th.matmul(x_a, contexts.unsqueeze(-1)).squeeze(-1)
        values = self.value_branch_2(x + x_a_out)
        # values = self.value_branch_2(x_a_out)

        return th.reshape(values, [-1])
    
    @override(ModelV2)
    def metrics(self):
        return {
        }

