# Sentinel-Isolate Screening Across Fungal and Oomycete Crop Pathosystems

Analysis code, configuration parameters, and precomputed summary tables supporting the manuscript:

> **Scale- and Population-Aware Sentinel-Isolate Selection for Crop Disease Resistance Screening Across Fungal and Oomycete Pathosystems**  
> Seunghyun Lim, Insuck Baek, Rakesh K. Upadhyay, Minhyeok Cha, Moon S. Kim, Lyndel W. Meinhardt, and Ezekiel Ahn

## Overview

This repository evaluates how sentinel-isolate panel design affects out-of-population top-$k$ host selection regret across three public-domain (CC0-1.0) multi-host plant pathology datasets:

1. ***Phytophthora sojae* – Soybean** (Hebb et al. 2022, *Phytopathology*; Dryad dataset DOI: `10.5061/dryad.kwh70rz1d`, Zenodo mirror record `5484368`): 455 isolates from Indiana ($n = 115$), Kentucky ($n = 94$), and Ohio ($n = 246$) across 62 unique state-by-location fields and 15 *Rps* differential lines (plus Williams check), 14 Illinois binary-only isolates, 34 repeated isolate records, and 19 diploid SSR loci ($n = 215$).
2. ***Zymoseptoria tritici* – Wheat** (Dutta et al. 2021a, *Evol. Appl.*; Dryad dataset DOI: `10.5061/dryad.j3tx95x9m`, Zenodo mirror record `4110942`): 145 isolates across four geographic regions (Australia 27, Israel 30, joint Oregon 56 [`Oregon.R` 26 + `Oregon.S` 30], Switzerland 32) and 12 wheat cultivars scored across 11,019 greenhouse seedling leaves for leaf necrosis (PLACL) and reproduction (pycnidia/cm²).
3. ***Botrytis cinerea* – Seven Eudicot Crops** (Caseys et al. 2021, *G3*; Dryad dataset DOI: `10.25338/B8Z91N`, Zenodo mirror record `4768042`): 98 isolates (96–98 complete per crop) across 84 host genotypes in seven crop species, 42,200 quality-filtered detached leaves with nested experimental blocks, and 271,749 biallelic SNPs ($n = 93$).

## Repository Structure

```text
sentinel-isolate-screening/
├── config.json                        # Random seeds, panel sizes, and simulation parameters
├── LICENSE                            # MIT License
├── README.md                          # Overview and reproduction instructions
├── requirements.txt                   # Python package dependencies
├── run_all.py                         # End-to-end analysis runner
├── scripts/
│   ├── fetch_sources.py               # Downloads CC0-1.0 Dryad mirrors from Zenodo with SHA-256 verification
│   ├── 01_harmonize_datasets.py       # Dataset harmonization and summary checks
│   ├── 02_sentinel_evaluation.py      # Out-of-population benchmarks, 2x2 scoring, genomics, spatial, leaf allocation
│   └── 03_simulation_models.py        # 200-breeding-line simulations and observation-model comparisons
└── results/                           # Summary CSV tables (20 files) and MAIN_TABLES.md
```

## Installation and Reproduction

Requires Python $\ge 3.10$.

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` downloads the three CC0-1.0 Dryad datasets (from their Zenodo archive mirrors) into `sources/` (skipping files already present and verified by SHA-256), writes harmonized intermediate tables to `data/processed/`, and regenerates all summary CSV tables in `results/`.

## Licensing

Analysis scripts and summary tables in this repository are released under the MIT License (`LICENSE`). Raw source datasets are not redistributed here; `scripts/fetch_sources.py` retrieves their mirrored files from Zenodo under their original Dryad CC0-1.0 dedications (`10.5061/dryad.kwh70rz1d`, `10.5061/dryad.j3tx95x9m`, and `10.25338/B8Z91N`).

## Funding

Supported by USDA-ARS In-House Projects Nos. 8042-21220-258-000-D and 8042-21000-303-000-D.
