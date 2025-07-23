from pathlib import Path
import yaml

class DotConfig:
    def __init__(self, data: dict, config_file: Path):
        self.config_file = config_file
        for k, v in data.items():
            if isinstance(v, dict):
                v = DotConfig(v, config_file)
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def load_config(yaml_path: Path) -> DotConfig:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DotConfig(config_dict, yaml_path)

def flatten_dict(d):
    items = {}
    for k, v in d.items():
        new_key = k
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[new_key] = v
    return items

def load_flat_config(yaml_path: Path) -> DotConfig:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    flat_dict = flatten_dict(config_dict)
    return DotConfig(flat_dict, yaml_path)

module_path = Path(__file__).parent.parent
configs_dir = module_path / "configs"
config_file = configs_dir / "config.yaml"
cfg = None # load_flat_config(config_file)