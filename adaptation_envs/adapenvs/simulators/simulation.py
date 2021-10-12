import matplotlib.pyplot as plt
from numpy.core.shape_base import stack
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
import numpy as np
import torch as th
from PIL import Image, ImageDraw, ImageFont
import random
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm
from copy import deepcopy
from adapenvs.utils.context_sampler import TRANSFORMS

def step_using_policy(agent_fn, obs, preprocessor):
    actions = {}
    with th.no_grad():
        for agent_id, obs in obs.items():
            actions[agent_id] = agent_fn(agent_id).get_policy("policy_1").compute_actions([preprocessor.transform(obs)])[0][0]
    return actions
