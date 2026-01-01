from pathlib import Path
import joblib

# Adjust this if your artifacts folder name is different
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models"


def _artifact_path(name: str) -> Path:
    return ARTIFACT_DIR / name


def artifact_exists(name: str) -> bool:
    p = _artifact_path(name)
    # size guard prevents “exists but is only an LFS pointer” in many cases
    return p.exists() and p.is_file() and p.stat().st_size > 1024


def load_artifact(name: str):
    p = _artifact_path(name)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")

    # Peek first bytes to detect Git LFS pointer or other non-binary content
    with p.open("rb") as f:
        head = f.read(200)

    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"Artifact '{name}' is a Git LFS pointer, not the real binary.\n"
            f"Fix: store the real .joblib in the repo (no LFS), or download it at runtime "
            f"(e.g., from GitHub Releases / S3) into {ARTIFACT_DIR}."
        )

    # Optional: extra guard (tiny files often indicate corruption)
    if p.stat().st_size < 1024:
        raise RuntimeError(
            f"Artifact '{name}' is too small ({p.stat().st_size} bytes) and likely corrupted."
        )

    try:
        return joblib.load(p)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load artifact '{name}' from {p}.\n"
            f"Possible causes: corrupted file, wrong file type, or incompatible sklearn/joblib versions."
        ) from e
