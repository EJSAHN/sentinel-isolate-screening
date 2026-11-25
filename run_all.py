"""Run the complete sentinel-isolate screening analysis pipeline.

Steps executed:
  1. scripts/fetch_sources.py          Download and verify CC0-1.0 Zenodo archives (skipped if present)
  2. scripts/01_harmonize_datasets.py  Harmonize multi-host phenotypes, replicates, and marker matrices
  3. scripts/02_sentinel_evaluation.py Run out-of-population benchmarks across all three pathosystems
  4. scripts/03_simulation_models.py   Run 200-breeding-line simulations and observation-model comparisons
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "scripts" / "fetch_sources.py",
    ROOT / "scripts" / "01_harmonize_datasets.py",
    ROOT / "scripts" / "02_sentinel_evaluation.py",
    ROOT / "scripts" / "03_simulation_models.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n>>> Running {script.relative_to(ROOT)} ...", flush=True)
        res = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
        if res.returncode != 0:
            raise SystemExit(f"Stopped: {script.name} exited with status {res.returncode}")
    print("\n>>> Pipeline completed successfully.")


if __name__ == "__main__":
    main()
