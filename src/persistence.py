from pathlib import Path
import joblib

MODELS_DIR = Path("models")

def load_artifact(filename: str):
    path = MODELS_DIR / filename
    return joblib.load(path)

def artifact_exists(filename: str) -> bool:
    return (MODELS_DIR / filename).exists()
