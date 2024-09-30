REGISTRY = {}

from .state_encoder import state_encoder

REGISTRY["state_reward"] = state_encoder