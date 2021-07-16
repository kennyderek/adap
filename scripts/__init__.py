from adap.models.concat_model import ConcatModel
from adap.models.mult_model import MultModel
from ray.rllib.models import ModelCatalog


ModelCatalog.register_custom_model(
    "Concat", ConcatModel)
ModelCatalog.register_custom_model(
    "Mult", MultModel)

