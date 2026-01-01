# src/persistence.py
from pathlib import Path
import joblib

# Your repo: models/ folder (same level as src/)
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _artifact_path(name: str) -> Path:
    return ARTIFACT_DIR / name


def _is_git_lfs_pointer(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def artifact_exists(name: str) -> bool:
    p = _artifact_path(name)
    if not (p.exists() and p.is_file()):
        return False
    if p.stat().st_size < 1024:
        return False
    if _is_git_lfs_pointer(p):
        return False
    return True


def load_artifact(name: str):
    p = _artifact_path(name)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")

    if _is_git_lfs_pointer(p):
        raise RuntimeError(
            f"Artifact '{name}' is a Git LFS pointer, not the real binary. "
            f"Upload real .joblib to GitHub Releases and download at runtime (Option A)."
        )

    if p.stat().st_size < 1024:
        raise RuntimeError(f"Artifact '{name}' looks corrupted (too small: {p.stat().st_size} bytes).")

    try:
        return joblib.load(p)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load artifact '{name}' from {p}. "
            f"Possible causes: corrupted file or incompatible sklearn/joblib versions."
        ) from e
