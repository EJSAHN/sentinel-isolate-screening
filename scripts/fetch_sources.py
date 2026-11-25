"""Download and verify SHA-256 checksums for the three CC0-1.0 Zenodo source archives."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = (
    (
        "Botrytis_Caseys2021_zenodo4768042.zip",
        "https://zenodo.org/api/records/4768042/files/Caseys_2021.zip/content",
        "35c63ad44e53491ba0b68a72869c3668d98ac290f8471d5a806f12be5826dff9",
    ),
    (
        "Zymo_Dutta2021_zenodo4110942.csv",
        "https://zenodo.org/api/records/4110942/files/Dutta_et_al_2020_phenotypic_Dataset.csv/content",
        "a72ffb0a00c0bd47c49726523eac929a9b89e3291bc626da8e9793238dd1efd5",
    ),
    (
        "Zymo_Dutta2021_README.txt",
        "https://zenodo.org/api/records/4110942/files/README.txt/content",
        "b187b675265c319534494139d411e265777fd715fe7fd0844c7150cc7abf8b4a",
    ),
    (
        "Psojae_Hebb2022_pathotype_zenodo5484368.xlsx",
        "https://zenodo.org/api/records/5484368/files/Pathotype_Data_for_P._sojae_characterization_in_OH_IN_IL_KY.xlsx/content",
        "9d5eebd03c17b3705664abc342b98707a75b6ef8a3ce57dc71d09fc5f9e9f26f",
    ),
    (
        "Psojae_Hebb2022_genotype_zenodo5484368.xlsx",
        "https://zenodo.org/api/records/5484368/files/Genotype_Data_for_P._sojae_characterization_iin_OH_IN_IL_KYxlsx.xlsx/content",
        "fcc6a6469943b4388784eba546c28614d227d53ac9aab08d817358b8de640a2f",
    ),
)


def fetch_all(sources_dir: Path) -> dict[str, dict]:
    sources_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for fname, url, expected_sha in SOURCES:
        target = sources_dir / fname
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"SHA-256 mismatch for existing {target}: {actual} != {expected_sha}")
            print(f"[OK] Verified existing {fname} ({target.stat().st_size:,} bytes)")
        else:
            print(f"[DL] Fetching {fname} from {url} ...")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (scientific-reproducibility)"})
            with urlopen(req, timeout=180) as resp:
                data = resp.read()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_sha:
                raise ValueError(f"SHA-256 mismatch for downloaded {fname}: {actual} != {expected_sha}")
            target.write_bytes(data)
            print(f"[OK] Downloaded & verified {fname} ({len(data):,} bytes)")
        report[fname] = {"bytes": target.stat().st_size, "sha256": expected_sha, "verified": True}
    return report


if __name__ == "__main__":
    fetch_all(Path(__file__).resolve().parent.parent / "sources")
