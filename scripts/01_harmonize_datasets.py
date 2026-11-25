"""
Step 1: Verify CC0-1.0 source files and build clean harmonized Isolate x Host matrices,
replicate-level tables, isolate metadata, and pathogen genetic distance matrices for:
  1. Zymoseptoria tritici  (Dutta et al. 2021, zenodo.4110942, CC0-1.0)
  2. Phytophthora sojae    (Hebb et al. 2022,  zenodo.5484368, CC0-1.0)
  3. Botrytis cinerea      (Caseys et al. 2021, zenodo.4768042, CC0-1.0)

All paths are relative to the project root.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = PROJECT_ROOT / "sources"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_SHA = {
    "Botrytis_Caseys2021_zenodo4768042.zip": "35c63ad44e53491ba0b68a72869c3668d98ac290f8471d5a806f12be5826dff9",
    "Zymo_Dutta2021_zenodo4110942.csv": "a72ffb0a00c0bd47c49726523eac929a9b89e3291bc626da8e9793238dd1efd5",
    "Psojae_Hebb2022_pathotype_zenodo5484368.xlsx": "9d5eebd03c17b3705664abc342b98707a75b6ef8a3ce57dc71d09fc5f9e9f26f",
    "Psojae_Hebb2022_genotype_zenodo5484368.xlsx": "fcc6a6469943b4388784eba546c28614d227d53ac9aab08d817358b8de640a2f",
}


def verify_hashes() -> dict[str, dict]:
    out = {}
    for fname, exp in EXPECTED_SHA.items():
        p = SOURCES_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing source: {p}")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != exp:
            raise ValueError(f"Hash mismatch on {fname}: {sha} != {exp}")
        out[fname] = {"bytes": p.stat().st_size, "sha256": sha}
    return out


# ---------------------------------------------------------------------------
# 1. Zymoseptoria tritici (Dutta et al. 2021)
# ---------------------------------------------------------------------------
def harmonize_zymoseptoria() -> dict:
    df = pd.read_csv(SOURCES_DIR / "Zymo_Dutta2021_zenodo4110942.csv")
    df = df.rename(columns={
        "Population": "population",
        "Isolate": "isolate_id",
        "Host": "host_id",
        "Replication": "rep_block",
        "Virulence (PLACL)": "placl",
        "Reproduction (pycnidia per cm2 of lesion)": "pycnidia_cm2",
    })
    df["isolate_id"] = df["isolate_id"].astype(str).str.strip()
    df["host_id"] = df["host_id"].astype(str).str.strip()
    df["population"] = df["population"].astype(str).str.strip()
    df["log1p_pycnidia"] = np.log1p(df["pycnidia_cm2"].clip(lower=0.0))

    # Replicate-level file
    df.to_csv(PROC_DIR / "zymo_replicates.csv", index=False)

    # Cell-mean Isolate x Host matrix (both PLACL and Pycnidia, plus log1p pycnidia)
    cell = (
        df.groupby(["isolate_id", "population", "host_id"], as_index=False)
        .agg(
            n_leaves=("placl", "size"),
            placl_mean=("placl", "mean"),
            placl_sd=("placl", "std"),
            pycnidia_mean=("pycnidia_cm2", "mean"),
            pycnidia_sd=("pycnidia_cm2", "std"),
            log1p_pycnidia_mean=("log1p_pycnidia", "mean"),
        )
    )
    cell.to_csv(PROC_DIR / "zymo_cell_means.csv", index=False)

    iso_meta = (
        cell.groupby(["isolate_id", "population"], as_index=False)
        .agg(
            n_hosts=("host_id", "nunique"),
            mean_placl=("placl_mean", "mean"),
            mean_pycnidia=("pycnidia_mean", "mean"),
            mean_log1p_pycnidia=("log1p_pycnidia_mean", "mean"),
        )
    )
    iso_meta.to_csv(PROC_DIR / "zymo_isolates.csv", index=False)

    # Verify each isolate belongs to exactly 1 population
    assert iso_meta["isolate_id"].nunique() == len(iso_meta), "Isolate spans multiple populations"

    return {
        "n_replicates": int(len(df)),
        "n_isolates": int(iso_meta["isolate_id"].nunique()),
        "n_hosts": int(cell["host_id"].nunique()),
        "n_cells": int(len(cell)),
        "completeness_pct": round(100.0 * len(cell) / (iso_meta["isolate_id"].nunique() * cell["host_id"].nunique()), 2),
        "populations": iso_meta["population"].value_counts().sort_index().to_dict(),
        "hosts": sorted(cell["host_id"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# 2. Phytophthora sojae (Hebb et al. 2022)
# ---------------------------------------------------------------------------
def norm_psojae_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def harmonize_psojae() -> dict:
    xl_path = SOURCES_DIR / "Psojae_Hebb2022_pathotype_zenodo5484368.xlsx"
    df_bin = pd.read_excel(xl_path, sheet_name="binary data sheet")
    df_gt = pd.read_excel(SOURCES_DIR / "Psojae_Hebb2022_genotype_zenodo5484368.xlsx", sheet_name="GenAlEx Allele Score")

    # Identify the 16 differential column pairs: (D1, rps (D1)), (D2, Rps1a (D2)), ...
    d_cols = [c for c in df_bin.columns if re.fullmatch(r"D\d+", str(c).strip())]
    col_list = list(df_bin.columns)
    diff_specs = []
    for dc in d_cols:
        pos = col_list.index(dc)
        bin_col = col_list[pos + 1]
        gene_name = str(bin_col).replace(f"({dc})", "").strip()
        diff_specs.append((dc, bin_col, gene_name))

    # Audit duplicates & missing rows in binary data sheet
    df_bin["isolate_raw"] = df_bin["Isolate ID"].astype(str).str.strip()
    df_bin["state"] = df_bin["State"].astype(str).str.strip()
    df_bin["county"] = df_bin["County"].astype(str).str.strip()
    df_bin["location_id"] = df_bin["Location ID"].astype(str).str.strip()

    # Detailed reconciliation of raw rows (including 34 repeated isolate IDs & 14 IL binary-only rows)
    dup_iso_rows = int(df_bin.duplicated(subset=["isolate_raw"]).sum())
    dup_ids = set(df_bin.loc[df_bin["isolate_raw"].duplicated(keep=False), "isolate_raw"].unique())

    # Export 34 paired repeated isolate IDs (Entry 1 vs Entry 2) for repeatability & binary-flip analysis
    rep_pair_rows = []
    for iso in sorted(dup_ids):
        grp = df_bin[df_bin["isolate_raw"] == iso]
        if len(grp) != 2:
            continue
        r1, r2 = grp.iloc[0], grp.iloc[1]
        for dc, bcol, gene in diff_specs:
            m1 = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(r1[dc])) if pd.notna(r1[dc]) else None
            m2 = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(r2[dc])) if pd.notna(r2[dc]) else None
            if m1 and m2:
                k1, n1 = int(m1.group(1)), int(m1.group(2))
                k2, n2 = int(m2.group(1)), int(m2.group(2))
                if n1 > 0 and n2 > 0 and k1 <= n1 and k2 <= n2:
                    p1, p2 = 100.0 * k1 / n1, 100.0 * k2 / n2
                    rep_pair_rows.append({
                        "isolate_id": iso,
                        "state": r1["state"],
                        "host_id": f"{dc}_{gene}",
                        "line_code": dc,
                        "rps_gene": gene,
                        "k_run1": k1, "n_run1": n1, "pct_run1": p1, "bin70_run1": int(p1 >= 70.0 - 1e-9),
                        "k_run2": k2, "n_run2": n2, "pct_run2": p2, "bin70_run2": int(p2 >= 70.0 - 1e-9),
                        "same_octal_sheet": int(str(r1.get("Octal Representation", "")) == str(r2.get("Octal Representation", ""))),
                    })
    pd.DataFrame(rep_pair_rows).to_csv(PROC_DIR / "psojae_replicates_34pairs.csv", index=False)

    # Audit raw sheet binary vs k/n >= 70% mismatches across all 515 raw rows
    raw_mis_dup = 0
    raw_mis_65_70 = 0
    raw_mis_other = 0
    for _, r in df_bin.iterrows():
        for dc, bcol, _ in diff_specs:
            if pd.isna(r[dc]) or pd.isna(r[bcol]):
                continue
            m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(r[dc]))
            if m and str(r[bcol]).strip() in ("0", "1", "0.0", "1.0"):
                k, n = int(m.group(1)), int(m.group(2))
                if n > 0 and k <= n:
                    pct = 100.0 * k / n
                    bi = int(float(r[bcol]))
                    ge70 = 1 if pct >= 70.0 - 1e-9 else 0
                    if bi != ge70:
                        if r["isolate_raw"] in dup_ids:
                            raw_mis_dup += 1
                        elif 65.0 <= pct < 70.0 and bi == 1:
                            raw_mis_65_70 += 1
                        else:
                            raw_mis_other += 1

    # Export IL binary-only isolates transparently (14 rows lacking raw k/n counts)
    il_rows = []
    for _, r in df_bin[df_bin["state"] == "IL"].iterrows():
        for dc, bcol, gene in diff_specs:
            bv = r[bcol]
            b_clean = int(float(bv)) if (pd.notna(bv) and str(bv).strip() in ("0", "1", "0.0", "1.0")) else np.nan
            il_rows.append({
                "isolate_id": r["isolate_raw"],
                "state": "IL",
                "county": r["county"],
                "location_id": r["location_id"],
                "host_id": f"{dc}_{gene}",
                "line_code": dc,
                "rps_gene": gene,
                "binary_sheet": b_clean,
            })
    pd.DataFrame(il_rows).to_csv(PROC_DIR / "psojae_il_binary_only.csv", index=False)

    # Build BOTH exact binomial pooled consensus (sum k / sum n for 34 replicated isolates; PRIMARY)
    # and first-run-only deduplication (SENSITIVITY)
    long_rows = []
    first_run_rows = []
    parse_issues = 0
    for iso, grp in df_bin.groupby("isolate_raw", sort=True):
        meta = grp.iloc[0]
        for dc, bcol, gene in diff_specs:
            ks, ns, bins_sheet = [], [], []
            for _, r in grp.iterrows():
                raw_val = r[dc]
                bin_val = r[bcol]
                if pd.notna(raw_val):
                    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(raw_val))
                    if m:
                        kv, nv = int(m.group(1)), int(m.group(2))
                        if nv > 0 and kv <= nv:
                            ks.append(kv)
                            ns.append(nv)
                        else:
                            parse_issues += 1
                    else:
                        parse_issues += 1
                if pd.notna(bin_val) and str(bin_val).strip() in ("0", "1", "0.0", "1.0"):
                    bins_sheet.append(int(float(bin_val)))

            if ks:
                k_pool, n_pool = int(sum(ks)), int(sum(ns))
                pct_pool = 100.0 * k_pool / n_pool
                b_from_pool = 1 if pct_pool >= 70.0 - 1e-9 else 0
                k_first, n_first = int(ks[0]), int(ns[0])
                pct_first = 100.0 * k_first / n_first
                b_from_first = 1 if pct_first >= 70.0 - 1e-9 else 0
            else:
                k_pool, n_pool, pct_pool, b_from_pool = np.nan, np.nan, np.nan, np.nan
                k_first, n_first, pct_first, b_from_first = np.nan, np.nan, np.nan, np.nan

            b_sheet = bins_sheet[0] if bins_sheet else b_from_pool
            base = {
                "isolate_id": iso,
                "state": meta["state"],
                "county": meta["county"],
                "location_id": meta["location_id"],
                "host_id": f"{dc}_{gene}",
                "line_code": dc,
                "rps_gene": gene,
                "n_runs_pooled": int(len(ks)),
            }
            long_rows.append({
                **base,
                "k_susc": k_pool,
                "n_plants": n_pool,
                "pct_susc": pct_pool,
                "binary_70": b_sheet,
                "binary_from_count": b_from_pool,
            })
            first_run_rows.append({
                **base,
                "k_susc": k_first,
                "n_plants": n_first,
                "pct_susc": pct_first,
                "binary_70": b_sheet,
                "binary_from_count": b_from_first,
            })

    long_df = pd.DataFrame(long_rows)
    first_df = pd.DataFrame(first_run_rows)
    valid_both = long_df.dropna(subset=["binary_70", "binary_from_count"])
    bin_mismatch = int((valid_both["binary_70"] != valid_both["binary_from_count"]).sum())

    # Filter to isolates with complete valid raw counts across all 16 differential lines
    iso_comp = long_df.groupby("isolate_id")["pct_susc"].apply(lambda s: int(s.notna().sum()))
    complete_isos = set(iso_comp[iso_comp == len(diff_specs)].index)
    long_complete = long_df[long_df["isolate_id"].isin(complete_isos)].copy()
    first_complete = first_df[first_df["isolate_id"].isin(complete_isos)].copy()

    long_df.to_csv(PROC_DIR / "psojae_all_long.csv", index=False)
    long_complete.to_csv(PROC_DIR / "psojae_complete_long.csv", index=False)
    first_complete.to_csv(PROC_DIR / "psojae_first_run_complete_long.csv", index=False)

    df_dedup = df_bin.drop_duplicates(subset=["isolate_raw"], keep="first").reset_index(drop=True)
    dup_conflicts = int(len(dup_ids))

    # Match SSR genotypes by normalized alphanumeric ID
    df_gt["norm_id"] = df_gt["Isolate ID"].apply(norm_psojae_id)
    iso_meta = (
        long_complete.groupby(["isolate_id", "state", "county", "location_id"], as_index=False)
        .agg(
            n_hosts=("host_id", "nunique"),
            mean_pct_susc=("pct_susc", "mean"),
            pathotype_complexity=("binary_from_count", "sum"),
            rps_susceptible_check_pct=("pct_susc", lambda s: float(s.iloc[0])),
        )
    )
    # Exact susceptible check D1 (Williams, rps) value per isolate
    d1_map = long_complete[long_complete["line_code"] == "D1"].set_index("isolate_id")["pct_susc"].to_dict()
    iso_meta["d1_check_pct"] = iso_meta["isolate_id"].map(d1_map)
    iso_meta["norm_id"] = iso_meta["isolate_id"].apply(norm_psojae_id)

    gt_dedup = df_gt.drop_duplicates(subset=["norm_id"], keep="first").set_index("norm_id")
    matched_norms = sorted(set(iso_meta["norm_id"]).intersection(set(gt_dedup.index)))
    # Exact two-sheet reconciliation summary (binary data sheet vs perc.susc data sheet)
    df_pct_sheet = pd.read_excel(xl_path, sheet_name="perc.susc data sheet")
    pct_id_col = [c for c in df_pct_sheet.columns if "iso" in str(c).lower()][0]
    ids_bin_set = set(df_bin["Isolate ID"].dropna().astype(str).str.strip())
    ids_pct_set = set(df_pct_sheet[pct_id_col].dropna().astype(str).str.strip())
    ids_pct_set.discard("")
    sheet_recon = {
        "binary_sheet_rows": int(len(df_bin)),
        "binary_sheet_unique_ids": int(len(ids_bin_set)),
        "perc_sheet_rows": int(len(df_pct_sheet)),
        "perc_sheet_nonnull_unique_ids": int(len(ids_pct_set)),
        "shared_ids": int(len(ids_bin_set & ids_pct_set)),
        "binary_only_ids": int(len(ids_bin_set - ids_pct_set)),
        "perc_only_ids": int(len(ids_pct_set - ids_bin_set)),
    }

    # Add composite state::location_id so identical local codes in different states never collide
    iso_meta["field_id"] = iso_meta["state"].astype(str) + "::" + iso_meta["location_id"].astype(str)
    iso_meta["has_ssr"] = iso_meta["norm_id"].isin(matched_norms)
    iso_meta.to_csv(PROC_DIR / "psojae_isolates.csv", index=False)

    # Compute pairwise SSR genetic distance (proportion of shared alleles / Bruvo-like allele mismatch) on matched isolates
    ssr_cols = list(df_gt.columns[4:-1])  # 19 loci x 2 alleles = 38 allele cols
    matched_iso_df = iso_meta[iso_meta["has_ssr"]].sort_values("isolate_id").reset_index(drop=True)
    if len(matched_iso_df) > 1:
        sub_gt = gt_dedup.loc[matched_iso_df["norm_id"], ssr_cols].apply(pd.to_numeric, errors="coerce").values
        n_m = len(matched_iso_df)
        n_loci = len(ssr_cols) // 2
        dist = np.zeros((n_m, n_m), dtype=float)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                shared = 0.0
                valid = 0
                for l in range(n_loci):
                    a1, a2 = sub_gt[i, 2 * l], sub_gt[i, 2 * l + 1]
                    b1, b2 = sub_gt[j, 2 * l], sub_gt[j, 2 * l + 1]
                    if np.isnan(a1) or np.isnan(a2) or np.isnan(b1) or np.isnan(b2) or a1 == 0 or b1 == 0:
                        continue
                    valid += 1
                    # allele sharing at diploid locus (0, 0.5, or 1.0)
                    s_ab = max(
                        (1.0 if a1 == b1 else 0.0) + (1.0 if a2 == b2 else 0.0),
                        (1.0 if a1 == b2 else 0.0) + (1.0 if a2 == b1 else 0.0),
                    ) / 2.0
                    shared += s_ab
                d_val = 1.0 - (shared / valid) if valid > 0 else np.nan
                dist[i, j] = dist[j, i] = d_val
        dist_df = pd.DataFrame(dist, index=matched_iso_df["isolate_id"], columns=matched_iso_df["isolate_id"])
        dist_df.to_csv(PROC_DIR / "psojae_ssr_dist.csv")

    return {
        "raw_sheet_rows": int(len(df_bin)),
        "duplicate_isolate_rows": dup_iso_rows,
        "duplicate_count_conflicts": dup_conflicts,
        "unique_isolates_total": int(len(df_dedup)),
        "complete_16diff_isolates": int(len(iso_meta)),
        "n_hosts": int(len(diff_specs)),
        "n_complete_cells": int(len(long_complete)),
        "parse_issues": parse_issues,
        "pooled_binary_vs_sheet_mismatches": bin_mismatch,
        "raw_sheet_mismatches_total": raw_mis_dup + raw_mis_65_70 + raw_mis_other,
        "raw_sheet_mismatches_in_34dups": raw_mis_dup,
        "raw_sheet_mismatches_65_70_as_1": raw_mis_65_70,
        "raw_sheet_mismatches_other": raw_mis_other,
        "replicated_isolate_pairs": int(len(dup_ids)),
        "replicated_paired_cells": int(len(rep_pair_rows)),
        "il_binary_only_isolates": int(df_bin[df_bin["state"] == "IL"]["isolate_raw"].nunique()),
        "states_complete": iso_meta["state"].value_counts().sort_index().to_dict(),
        "n_counties_complete": int(iso_meta["county"].nunique()),
        "n_locations_complete": int(iso_meta["location_id"].nunique()),
        "n_fields_state_location": int(iso_meta["field_id"].nunique()),
        "fields_per_state": iso_meta.groupby("state")["field_id"].nunique().sort_index().to_dict(),
        "two_sheet_reconciliation": sheet_recon,
        "ssr_matched_complete_isolates": int(len(matched_iso_df)),
        "differentials": [f"{dc}_{g}" for dc, _, g in diff_specs],
    }


# ---------------------------------------------------------------------------
# 3. Botrytis cinerea (Caseys et al. 2021)
# ---------------------------------------------------------------------------
def norm_botrytis_id(s: str) -> str:
    s0 = str(s).strip()
    # Map e.g. '1.01.01' <-> '01_01_01' and '2.04.03' <-> '02_04_03'
    m = re.fullmatch(r"0?(\d+)[._]0?(\d+)[._]0?(\d+)", s0)
    if m:
        return f"{int(m.group(1)):02d}_{int(m.group(2)):02d}_{int(m.group(3)):02d}"
    return re.sub(r"[^a-z0-9]", "", s0.lower())


def harmonize_botrytis() -> dict:
    zpath = SOURCES_DIR / "Botrytis_Caseys2021_zenodo4768042.zip"
    with zipfile.ZipFile(zpath) as zf:
        ls_long = pd.read_csv(io.BytesIO(zf.read("Caseys_2021/Eudi_Ara_Lsmeans1_8sp_row_Geno_All_plot.txt")), sep="\t")
        iso_orig = pd.read_csv(io.BytesIO(zf.read("Caseys_2021/Isolate_Host_origin_CORRECTED.txt")), sep="\t")
        acc_list = pd.read_csv(io.BytesIO(zf.read("Caseys_2021/accession.list7sp.txt")), sep="\t")
        raw_rep = pd.read_csv(io.BytesIO(zf.read("Caseys_2021/Eudicot7sp_clean_median.txt")), sep=r"\s+", low_memory=False)
        snp_df = pd.read_csv(io.BytesIO(zf.read("Caseys_2021/BCinereaSNPdata.csv")), low_memory=False)

    # Clean LSmeans long table (8 species: 7 crop species x 12 accessions + Arabidopsis Col0/coi1)
    ls_long = ls_long.rename(columns={
        "Isolate": "isolate_id",
        "PGeno": "host_id",
        "Taxon": "crop_species",
        "Wi_Do": "domestication",
        "Order": "plant_order",
        "LsMeans": "lesion_lsmean",
    })
    ls_long["isolate_id"] = ls_long["isolate_id"].astype(str).str.strip()
    ls_long["host_id"] = ls_long["host_id"].astype(str).str.strip()
    ls_long["crop_species"] = ls_long["crop_species"].astype(str).str.strip()
    ls_long.to_csv(PROC_DIR / "botrytis_lsmeans_long.csv", index=False)

    # Raw replicate table (7 crop species, 42,200 leaves) - keep essential columns cleanly
    rep_keep = raw_rep[["IsolateID", "PlantGeno", "Taxon", "Domest", "Exp", "Rep", "Flat", "IndPlant", "Scale.LS"]].copy()
    rep_keep.columns = ["isolate_id", "host_id", "crop_species", "domestication", "exp", "rep", "flat", "ind_plant", "lesion_cm2"]
    rep_keep["isolate_id"] = rep_keep["isolate_id"].astype(str).str.strip()
    rep_keep["host_id"] = rep_keep["host_id"].astype(str).str.strip()
    rep_keep.to_csv(PROC_DIR / "botrytis_replicates_7sp.csv", index=False)

    # Isolate metadata
    iso_orig = iso_orig.rename(columns={"Isolate": "isolate_id", "Origin": "origin", "Host": "origin_host", "Group": "genetic_group", "GROUP01": "genetic_group_clean"})
    iso_orig["isolate_id"] = iso_orig["isolate_id"].astype(str).str.strip()
    iso_orig["norm_id"] = iso_orig["isolate_id"].apply(norm_botrytis_id)

    # Match SNP matrix columns (cols 3..97 -> 95 isolates) to phenotype isolate_id
    snp_iso_cols = list(snp_df.columns[3:])
    snp_norm_to_col = {norm_botrytis_id(c): c for c in snp_iso_cols}
    pheno_isos = sorted(ls_long["isolate_id"].unique().tolist())
    pheno_norm_to_id = {norm_botrytis_id(i): i for i in pheno_isos}

    matched_pairs = []
    for pid in pheno_isos:
        nid = norm_botrytis_id(pid)
        if nid in snp_norm_to_col:
            matched_pairs.append((pid, snp_norm_to_col[nid]))

    # Compute pairwise SNP p-distance matrix (271,749 biallelic 0/1 SNPs) via matrix multiplication
    matched_pids = [p for p, _ in matched_pairs]
    matched_cols = [c for _, c in matched_pairs]
    G = snp_df[matched_cols].to_numpy(dtype=np.float32)  # shape (271749, n_matched)
    valid = ~np.isnan(G)
    G0 = np.where(valid, G, 0.0).astype(np.float32)
    G1 = np.where(valid, 1.0 - G, 0.0).astype(np.float32)
    # mismatches between i and j = G0_i * G1_j + G1_i * G0_j
    n_valid = valid.astype(np.float32).T @ valid.astype(np.float32)
    n_diff = (G0.T @ G1) + (G1.T @ G0)
    snp_dist = np.divide(n_diff, n_valid, out=np.full_like(n_diff, np.nan), where=n_valid > 0)
    np.fill_diagonal(snp_dist, 0.0)
    snp_dist_df = pd.DataFrame(np.round(snp_dist, 6), index=matched_pids, columns=matched_pids)
    snp_dist_df.to_csv(PROC_DIR / "botrytis_snp_dist.csv")

    iso_orig["has_snp"] = iso_orig["isolate_id"].isin(set(matched_pids))
    iso_orig.to_csv(PROC_DIR / "botrytis_isolates.csv", index=False)

    # Summary stats per crop species
    crop_stats = {}
    for sp, grp in ls_long.groupby("crop_species"):
        n_g = int(grp["host_id"].nunique())
        n_i = int(grp["isolate_id"].nunique())
        non_na = int(grp["lesion_lsmean"].notna().sum())
        crop_stats[sp] = {
            "n_genotypes": n_g,
            "n_isolates": n_i,
            "non_na_cells": non_na,
            "completeness_pct": round(100.0 * non_na / (n_g * n_i), 2),
        }

    return {
        "n_raw_replicates_7sp": int(len(rep_keep)),
        "n_isolates_pheno": int(len(pheno_isos)),
        "n_hosts_total": int(ls_long["host_id"].nunique()),
        "n_crop_species": int(ls_long[ls_long["crop_species"] != "Athaliana"]["crop_species"].nunique()),
        "n_snp_loci": int(len(snp_df)),
        "n_snp_matched_isolates": int(len(matched_pids)),
        "genetic_groups": iso_orig["genetic_group_clean"].value_counts(dropna=False).to_dict(),
        "crop_species_summary": crop_stats,
    }


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    hashes = verify_hashes()
    print("[1/3] Harmonizing Zymoseptoria tritici (Dutta 2021)...", flush=True)
    z_info = harmonize_zymoseptoria()
    print("[2/3] Harmonizing Phytophthora sojae (Hebb 2022)...", flush=True)
    p_info = harmonize_psojae()
    print("[3/3] Harmonizing Botrytis cinerea (Caseys 2021, incl. 271K SNP distances)...", flush=True)
    b_info = harmonize_botrytis()

    audit = {
        "sources_sha256": hashes,
        "zymoseptoria_dutta2021": z_info,
        "psojae_hebb2022": p_info,
        "botrytis_caseys2021": b_info,
    }
    (PROC_DIR / "harmonization_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== HARMONIZATION COMPLETE ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
