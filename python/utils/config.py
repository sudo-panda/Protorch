from pathlib import Path
import yaml

class DotConfig:
    def __init__(self, data: dict, config_file: Path, flatten: bool = True):
        self.config_file = config_file
        self.data = data
        self.flat_data = None

        if not flatten:
            for k, v in data.items():
                if isinstance(v, dict):
                    v = DotConfig(v, config_file)
                setattr(self, k, v)
        else:
            self.flat_data = flatten_dict(data)
            for k, v in self.flat_data.items():
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.data[key] = value
        setattr(self, key, value)

    def save(self, path: Path):
        with open(path, 'w') as f:
            yaml.dump(self.data, f)


def load_config(yaml_path: Path, flatten: bool = True) -> DotConfig:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DotConfig(config_dict, yaml_path, flatten=flatten)

def flatten_dict(d):
    items = {}
    for k, v in d.items():
        new_key = k
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[new_key] = v
    return items

module_path = Path(__file__).parent.parent
configs_dir = module_path / "configs"
config_file = configs_dir / "config.yaml"
cfg = None # load_flat_config(config_file)