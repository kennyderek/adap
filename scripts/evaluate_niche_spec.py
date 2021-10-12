from utils.eval import restore_model, playthrough_simple
import os
from adapenvs.farmworld.farmworld import Farmworld
import yaml
import numpy as np


from adap.models.concat_model import ConcatModel
from adap.models.mult_model import MultModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model(
    "ContextConcat", ConcatModel)
ModelCatalog.register_custom_model(
    "ContextMult", MultModel)
import ray
ray.init()


import logging
logging.basicConfig(filename='niche_data.log', level=logging.INFO)

base = "/Users/kderek/nipsbutt/niched/niche/"
train_conf_base = "/Users/kderek/projects/adap/configs/farmworld/train/"
env_conf_base = "/Users/kderek/projects/adap/configs/farmworld/niche_specialization_env.yaml"
with open(env_conf_base, 'r') as f:
    env_conf = yaml.load(f)

def parse_game_data(all_infos):
    lifetimes = []
    avt = []
    avc = []
    attacktropy = []
    for step_i in all_infos:
        for id, agent_info in step_i.items():
            if "lifetime" in agent_info:
                lifetimes.append(agent_info["lifetime"])
            if "avt" in agent_info:
                avt.append(agent_info["avt"])
            if "avc" in agent_info:
                avc.append(agent_info["avc"])
            if "c_t_attacktropy" in agent_info:
                attacktropy.append(agent_info["c_t_attacktropy"])
    return [np.mean(lifetimes), np.mean(avt), np.mean(avc), np.mean(attacktropy)]

for method_seed in sorted(os.listdir(base)):
    checkpoint = sorted([n for n in os.listdir(os.path.join(base, method_seed)) if "checkpoint" in n])[-1]

    restore_path = os.path.join(base, method_seed, checkpoint, "checkpoint-"+checkpoint.split("_")[-1][2:])

    algo = ""
    if "adap" in method_seed:
        algo = "adap"
        continue
    if "vanilla" in method_seed:
        algo = "vanilla"
        continue
    if "diayn" in method_seed:
        algo = "diayn"
        if "concat" in method_seed:
            continue

    model = "ContextMult"
    if "concat" in method_seed:
        model = "ContextConcat"

    agent = restore_model(restore_path, os.path.join(train_conf_base, algo+".yaml"), env_conf_base, Farmworld, model=model)
    
    all_parsed = []
    for i in range(0, 100):
        game_data = playthrough_simple(lambda id: agent, Farmworld, env_conf)
        parsed = parse_game_data(game_data)
        all_parsed.append(parsed)
    
    line = method_seed + "," + checkpoint
    for i in np.array(all_parsed).mean(axis=0):
        line += ",{}".format(i)
    logging.info(line)