from .clevr_defaults import _C as clevr_cfg
from yacs.config import CfgNode


def get_default_cfg(mode="clevr"):
    if mode == "clevr":
        return clevr_cfg
    raise NotImplementedError()


def config_to_dict(cfg):
    local_dict = {}
    for key, value in cfg.items():
        if type(value) != CfgNode:
            local_dict[key] = value
        else:
            local_dict[key] = config_to_dict(value)
    return local_dict


def config_to_wandb(cfg):
    """Convert config to wandb-compatible format"""
    def _config_to_wandb(cfg, local_dict, parent_str):
        for key, value in cfg.items():
            full_key = "{}.{}".format(parent_str, key)
            if type(value) != CfgNode:
                local_dict[full_key] = value
            else:
                _config_to_wandb(value, local_dict, full_key)

    local_dict = {}
    for key, value in cfg.items():
        if type(value) != CfgNode:
            local_dict[key] = value
        else:
            _config_to_wandb(value, local_dict, key)
    return local_dict
