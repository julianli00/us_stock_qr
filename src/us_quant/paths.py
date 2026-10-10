from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts" / "current"
REPORT_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [
        CONFIG_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACT_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
