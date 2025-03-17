import os
import yaml
from omegaconf import OmegaConf

# Initialize Hydra and load the config with all references resolved
from hydra import initialize, compose

with initialize(version_base=None, config_path="conf"):
    # This will load the main config and all referenced configs
    cfg = compose(config_name="config")
    
    # Convert to a regular dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Write to params.yaml
    with open('params.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)