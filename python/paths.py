from pathlib import Path

file_path = Path(__file__).resolve()
Protorch_python_path = file_path.parent
Protorch_path = Protorch_python_path.parent
top_level_path = Protorch_path.parent
GraMI_path = Protorch_python_path / "GraMI"