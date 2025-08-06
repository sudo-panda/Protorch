from pathlib import Path

file_path = Path(__file__).resolve()
utils_path = file_path.parent
Protorch_python_path = utils_path.parent
Protorch_path = Protorch_python_path.parent
top_level_path = Protorch_path.parent
models_path = Protorch_python_path / "models"
runs_dir = Protorch_python_path / "runs"
GraMI_path = models_path / "GraMI"
Devmap_path = models_path / "Devmap"

def get_data_paths(dataset):
    datadir_name_map = {
        "devmap": ["devmap", "heterodatas"],
        "devmap-amd": ["devmap", "heterodatas-amd"],
        "hecbench": ["HecBench", "heterodatas"]
    }

    datasetdir_names = datadir_name_map.get(dataset.lower(), [dataset, "heterodatas"])
    dataset_dir = top_level_path / datasetdir_names[0]
    assert dataset_dir.exists() and dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist. Please check the dataset name: {dataset}"
    data_path = dataset_dir / datasetdir_names[1]
    assert data_path.exists() and data_path.is_dir(), f"Data path {data_path} does not exist. Please check the data directory."
    csv_file  = data_path / "datapoints.csv"
    assert csv_file.exists(), f"CSV file {csv_file} does not exist. Please check the file path."

    return dataset_dir, data_path, csv_file