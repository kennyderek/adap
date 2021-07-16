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

class ConcatModel(BaseModel):
    """PyTorch version of the CustomLossModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

        self.agent_branch = nn.Sequential(
            nn.Linear(self.get_input_size_inluding_ctx(), self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim), # added recently
            nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), # added recently
            # self.value_nonlinearity(),
            nn.Linear(self.hidden_dim, num_outputs)
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self.get_input_size_inluding_ctx(), self.hidden_dim),
            self.value_nonlinearity(),
            nn.Linear(self.hidden_dim, self.hidden_dim), # added recently
            self.value_nonlinearity(),
            # nn.Linear(self.hidden_dim, self.hidden_dim), # added recently
            # self.value_nonlinearity(),
            nn.Linear(self.hidden_dim, 1)
        )

        if self.use_mi_discrim:
            self.initialize_mi_discrim()

        self._last_obs_seen = None
        # self.rew_mod_mean = nn.Parameter(th.tensor(0), requires_grad=False)
        # self.rew_mod_sig = nn.Parameter(th.tensor(1), requires_grad=False)

    def get_input_size_excluding_ctx(self):
        return self.obs_space_size - self.context_size

    def get_input_size_inluding_ctx(self):
        return self.obs_space_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # TODO: test if a positively weighted linear encoder with no biases can help reshape the context distribution, positively!
        # this could help in situations like the uneven field experiment!
        observations = input_dict["obs_flat"].float()
        self._last_obs_seen = observations

        logits = self.agent_branch(observations)

        if self.clamp_mu_logits:
            logits = self.softsign_mu(logits)
        
        return logits, state

    @override(ModelV2)
    def value_function(self):
        values = self.value_branch(self._last_obs_seen)
        return th.reshape(values, [-1])
    
    @override(ModelV2)
    def metrics(self):
        return {
        }

