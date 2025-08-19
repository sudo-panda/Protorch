from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Optional, Union, Any

@dataclass
class TrainConfig:
    dataset: str
    model_name: str
    device: str = "cuda"
    seed: int = 42
    epochs: int = 1000
    world_size: Optional[int] = None
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    train_from_checkpoint: Union[bool, str] = False
    optimizer: str = "AdamW"
    scheduler: Optional[dict] = None
    scaler: Optional[dict] = None
    loss_config: Optional[dict] = None
    extra_config: dict = field(default_factory=dict)

    # For easy access only, is not setable
    training_mode: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainConfig":
        train_config = config_dict.get('train', {})
        del config_dict['train']
        config_dict = config_dict | train_config
        if 'test' in config_dict:
            del config_dict['test']

        assert "training_mode" not in config_dict, "Config is trying to set training mode, its an internal parameter"

        # Check for extra keys added during the run
        extra_keys = set(config_dict.keys()) - set(cls.__dataclass_fields__.keys())
        if extra_keys:
            extra_dict = {k: config_dict[k] for k in extra_keys}
            for key in extra_keys:
                del config_dict[key]
            cfg = cls(**config_dict)
            for k, v in extra_dict.items():
                cfg[k] = v
            return cfg
        else:
            return cls(**config_dict)

    def save(self, path: Path):
        default_keys = ['device', 'model_name', 'seed']
        train_cfg = {k: v for k, v in self.__dict__.items() if k not in default_keys and k != "training_mode"}
        config_dict = {'train': train_cfg}
        for key in default_keys:
            config_dict[key] = self.__dict__[key]
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)

    def dumps(self, **kwargs) -> str:
        default_keys = ['device', 'model_name', 'seed']
        train_cfg = {k: v for k, v in self.__dict__.items() if k not in default_keys and k != "training_mode"}
        config_dict = {'train': train_cfg}
        for key in default_keys:
            config_dict[key] = self.__dict__[key]

        return yaml.dump(config_dict, **kwargs)

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

@dataclass
class TestConfig:
    dataset: str
    model_name: str
    device: str = "cuda"
    batch_size: int = 16
    checkpoint_dir: Optional[str] = None
    checkpoint_file_name: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TestConfig":
        test_config = config_dict.get('test', {})
        del config_dict['test']
        config_dict = config_dict | test_config
        if 'train' in config_dict:
            del config_dict['train']

        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            config_dict['checkpoint_dir'] = checkpoint.get('dir', None)
            config_dict['checkpoint_file_name'] = checkpoint.get('file_name', None)
            del config_dict['checkpoint']

        return cls(**config_dict)

    def save(self, path: Union[Path, str]):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

    def dumps(self) -> str:
        return yaml.dump(self.__dict__)

def load_config(yaml_path: Path, train: bool = True) -> Union[TrainConfig, TestConfig]:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if train:
        return TrainConfig.from_dict(config_dict)
    else:
        return TestConfig.from_dict(config_dict)

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