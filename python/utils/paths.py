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