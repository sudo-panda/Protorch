from pathlib import Path
from typing import Union

file_path = Path(__file__).resolve()
utils_path = file_path.parent
Protorch_python_path = utils_path.parent
Protorch_path = Protorch_python_path.parent
top_level_path = Protorch_path.parent
models_path = Protorch_python_path / "models"
configs_dir = Protorch_python_path / "configs"
runs_dir = Path("/p/vast1/kundu1/protorch") / "runs"
GraMI_path = models_path / "GraMI"
Devmap_path = models_path / "Devmap"

def get_data_paths(dataset: str) -> tuple[Path, Union[Path, dict[str, Path]], Union[Path, dict[str, Path]]]:
    path_map = {
        "devmap": {
            "dataset": "/p/vast1/kundu1/protorch/devmap", 
            "data": "heterodatas"
        },
        "devmap-amd": {
            "dataset": "/p/vast1/kundu1/protorch/devmap", 
            "data": "heterodatas-amd"
        },
        "hecbench": {
            "dataset": "HecBench", 
            "data": "heterodatas"
        },
        "vecparams-x86_64-O3-best_VF_IF": {
            "dataset": "/p/vast1/kundu1/protorch/NeuroVectorizer", 
            "data": {
                "train": "heterodatas_toss_4_x86_64_ib_cray_O3/training_data",
                "test": "heterodatas_toss_4_x86_64_ib_cray_O3/tests"
            },
            "csv": "best_VF_IF.csv"
        },
        "vecparams-x86_64-O0-best_VF_IF": {
            "dataset": "/p/vast1/kundu1/protorch/NeuroVectorizer", 
            "data": {
                "train": "heterodatas_toss_4_x86_64_ib_cray_O0/training_data",
                "test": "heterodatas_toss_4_x86_64_ib_cray_O0/tests"
            },
            "csv": "best_VF_IF.csv"
        },
        "vecparams-x86_64-O3-wgted_VF_IF": {
            "dataset": "/p/vast1/kundu1/protorch/NeuroVectorizer", 
            "data": {
                "train": "heterodatas_toss_4_x86_64_ib_cray_O3/training_data",
                "test": "heterodatas_toss_4_x86_64_ib_cray_O3/tests"
            },
            "csv": "wgted_VF_IF.csv"
        },
        "vecparams-x86_64-O0-wgted_VF_IF": {
            "dataset": "/p/vast1/kundu1/protorch/NeuroVectorizer", 
            "data": {
                "train": "heterodatas_toss_4_x86_64_ib_cray_O0/training_data",
                "test": "heterodatas_toss_4_x86_64_ib_cray_O0/tests"
            },
            "csv": "wgted_VF_IF.csv"
        },
    }

    path_map = {k.lower(): v for k, v in path_map.items()}

    assert dataset.lower() in path_map, f"Unknown dataset: {dataset.lower()}! Check path_map in {__file__}"
    paths = path_map[dataset.lower()]
    dataset_dir = top_level_path / paths["dataset"]
    assert dataset_dir.exists() and dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist. Please check the dataset name: {dataset}"
    data_rel_dir = paths["data"]
    if isinstance(data_rel_dir, dict):
        data_path = {key: (dataset_dir / dr) for key, dr in data_rel_dir.items()}

        for dp in data_path.values():
            assert dp.exists() and dp.is_dir(), f"Data path {dp} does not exist. Please check the data directory."
    else:
        data_path = dataset_dir / data_rel_dir
        assert data_path.exists() and data_path.is_dir(), f"Data path {data_path} does not exist. Please check the data directory."

    if "csv" in paths:
        csv_file_name = paths["csv"]
    else:
        csv_file_name = "datapoints.csv"
    
    if isinstance(data_path, dict):
        csv_file = {key: (dp / csv_file_name) for key, dp in data_path.items()}
        for file in csv_file.values():
            assert file.exists(), f"CSV file {file} does not exist. Please check the file path."
    else:
        csv_file = data_path / csv_file_name
        assert csv_file.exists(), f"CSV file {csv_file} does not exist. Please check the file path."

    return dataset_dir, data_path, csv_file