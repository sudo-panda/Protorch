from pathlib import Path

file_path = Path(__file__).resolve()
Protorch_python_path = file_path.parent
Protorch_path = Protorch_python_path.parent
top_level_path = Protorch_path.parent
models_path = Protorch_python_path / "models"
GraMI_path = models_path / "GraMI"
GraMI_wgts_dir = GraMI_path / "weights"
Devmap_path = models_path / "Devmap"
Devmap_wgts_dir = Devmap_path / "weights"

GraMI_wgts_dir.mkdir(exist_ok=True)
Devmap_wgts_dir.mkdir(exist_ok=True)