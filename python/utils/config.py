import yaml
from pathlib import Path


script_path = Path(__file__)
module_path = script_path.parent.parent.resolve()
config_path = module_path / "config.yaml"

def get_config():
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return config_dict

config = get_config()

device = config["device"]
epochs =  config["train"]["epochs"]
train_from_checkpoint = config["train"]["train_from_checkpoint"]
lr = config["train"]["learning_rate"]
decay = config["train"]["weight_decay"]
batch_size = config["train"]["batch_size"]
dataset = config["train"]["dataset"]

if "world_size" in config["train"]:
    world_size = config["train"]["world_size"]