"""
Step 2: Out-of-Population Measurement-Scale, Item-Response, Threshold, Genomic, and Spatial
Sentinel-Isolate Evaluation across three CC0-1.0 crop pathosystems:
  1. Phytophthora sojae  (Hebb et al. 2022,  n=455 complete in 3 states + 14 IL binary + 34 rep pairs + 215 SSRs)
  2. Zymoseptoria tritici (Dutta et al. 2021, n=145 isolates in 4 regions [5 pops] x 12 cultivars, 11,019 seedling leaves)
  3. Botrytis cinerea     (Caseys et al. 2021, n=98 isolates x 84 genotypes across 7 crops, 42,200 detached leaves, 271K SNPs)

Analytical components:
  - Shared sampled panels and tie-breaking draws for soybean scoring and wheat cross-endpoint comparisons;
    pool-specific draws for comparisons across distinct candidate pools.
  - Fold-level primary inference (fold_level_strategies.csv + exact sign-flip tests across folds);
    Monte Carlo intervals explicitly labeled as subset-sampling precision.
  - Scale sensitivity (raw vs logit/log scores & truth) and Fisher-Information sentinel rule (InfoOpt).
  - Model-based host ranking (binomial-logit / logit-normal / log-normal) vs arithmetic averaging.
  - Exact 2 x 2 truth-by-scoring benchmark in P. sojae (continuous mortality vs virulent proportion >=70%),
    intermediate-efficacy Rps subset, and full Illinois metrics.
  - Binomial expected flip rate, Beta-Binomial MLE overdispersion, and host-adjusted ICC on 34 P. sojae pairs.
  - Split-host (7 vs 8 Rps lines, 50 splits) virulence proxy for P. sojae.
  - Joint Oregon holdout (4 geographic regions) + 5-population sensitivity for Z. tritici cross-endpoint & virulence.
  - Strict equal-m SingleField sampling (only fields with >= m isolates; realized_m == m verified).
  - True LOSO SSR genomic covering in P. sojae + Partial Mantel conditioned on field/state + strict ancestry-letter holdout in B. cinerea.
  - Blocked mixed-model variance refit (Exp + Flat + IndPlant1) for B. cinerea + replacement-free fixed-budget resampling.

All paths are relative to the project root.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, gammaln, logit
from scipy.stats import binom, chi2, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
RES = PROJECT_ROOT / "results"
CFG = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))

SEED = int(CFG["seed"])
M_GRID = list(CFG["m_grid"])
M_VIR = list(CFG["m_virulence_grid"])
M_GEN = list(CFG["m_genomic_grid"])


# ---------------------------------------------------------------------------
# Fast exact metrics, CRN helpers, & fold-level exact sign-flip inference
# ---------------------------------------------------------------------------
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _avg_ranks(x)
    ry = _avg_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else 0.0


def eval_selection(scores: np.ndarray, truth: np.ndarray, k: int, jitter: np.ndarray) -> dict[str, float]:
    """Lower value = stronger resistance. Uses pre-generated CRN jitter for exact tie-breaking."""
    sel = np.argsort(scores + jitter)[:k]
    orc = np.argsort(truth)[:k]
    regret = float(np.mean(truth[sel]) - np.mean(truth[orc]))
    t_range = float(np.max(truth) - np.min(truth))
    norm_regret = regret / t_range if t_range > 1e-12 else 0.0
    rho = fast_spearman(scores, truth)
    med = float(np.median(truth))
    severe_miss = 1.0 if np.any(truth[sel] > med + 1e-12) else 0.0
    return {"regret": regret, "norm_regret": norm_regret, "rho": rho, "severe_miss": severe_miss}


def exact_sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation P-value across K folds (K <= 16)."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    K = len(d)
    if K == 0:
        return np.nan
    obs = abs(float(np.mean(d)))
    if obs < 1e-12:
        return 1.0
    if K <= 14:
        cnt = 0
        total = 1 << K
        for signs in itertools.product((-1.0, 1.0), repeat=K):
            if abs(float(np.dot(d, signs)) / K) >= obs - 1e-12:
                cnt += 1
        return cnt / total
    rng = np.random.default_rng(SEED + 99)
    S = rng.choice([-1.0, 1.0], size=(4000, K))
    means = np.abs((S @ d) / K)
    return float((np.sum(means >= obs - 1e-12) + 1.0) / 4001.0)


def mc_precision_ci(arr: np.ndarray, rng: np.random.Generator, n_boot: int = 400) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= 1 or np.allclose(arr, arr[0]):
        v = float(np.mean(arr)) if len(arr) else np.nan
        return v, v
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def logit_bounded(p_pct: np.ndarray, eps: float = 0.025) -> np.ndarray:
    """Variance-stabilized logit transform for bounded [0, 100]% scores."""
    p = np.clip(p_pct / 100.0, 0.0, 1.0)
    p_s = (p + eps) / (1.0 + 2.0 * eps)
    return np.log(p_s / (1.0 - p_s))


def log_positive(y: np.ndarray, eps: float = 0.05) -> np.ndarray:
    return np.log(np.clip(y, 0.0, None) + eps)


def variance_decomp_2way(mat: np.ndarray) -> dict[str, float]:
    g = np.nanmean(mat)
    ie = np.nanmean(mat, axis=1, keepdims=True) - g
    he = np.nanmean(mat, axis=0, keepdims=True) - g
    inter = mat - g - ie - he
    ss_h = float(np.nansum(np.broadcast_to(he, mat.shape) ** 2))
    ss_i = float(np.nansum(np.broadcast_to(ie, mat.shape) ** 2))
    ss_ih = float(np.nansum(inter ** 2))
    tot = ss_h + ss_i + ss_ih
    return {
        "host_pct": round(100.0 * ss_h / tot, 2),
        "isolate_pct": round(100.0 * ss_i / tot, 2),
        "ih_pct": round(100.0 * ss_ih / tot, 2),
        "ih_to_h": round(ss_ih / max(ss_h, 1e-12), 3),
    }


def variance_decomp_4way_exact(df_rep: pd.DataFrame, val_col: str, block_cols: list[str] | None = None) -> dict[str, float]:
    """Unbiased 4-way variance partitioning using weighted cell-size ANOVA (and optional block removal)."""
    sub = df_rep.dropna(subset=[val_col]).copy()
    if block_cols:
 # Exact Caseys et al. nested block decomposition:
 # Exp + Flat are crossed with Host, whereas IndPlant1 = Host-Exp-IndPlant is nested inside Host x Exp
 # Separating non-host blocks (Exp, Flat) BEFORE Host, and nested IndPlant1 AFTER Host preserves
 # true between-host variance while removing all within-host individual-plant and flat variation.
        y = sub[val_col].to_numpy(dtype=float)
        y_c = y - float(y.mean())
        cell_codes = pd.factorize(sub["isolate_id"].astype(str) + "||" + sub["host_id"].astype(str))[0]
        iso_codes = pd.factorize(sub["isolate_id"].astype(str))[0]
        host_codes = pd.factorize(sub["host_id"].astype(str))[0]
        ef_cols = [c for c in block_cols if c != "ind_plant1"]
        ef_codes = [pd.factorize(sub[c].astype(str))[0] for c in ef_cols]
        ind_codes = [pd.factorize(sub["ind_plant1"].astype(str))[0]] if "ind_plant1" in block_cols else []

        def residualize(vec: np.ndarray, grp_list: list[np.ndarray], n_iter: int = 30) -> np.ndarray:
            r = vec.copy()
            for _ in range(n_iter):
                for gc in grp_list:
                    n_g = int(gc.max()) + 1
                    sums = np.bincount(gc, weights=r, minlength=n_g)
                    cnts = np.bincount(gc, minlength=n_g)
                    means = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
                    r -= means[gc]
            return r

 # Exact orthogonal sequential SS decomposition (sums to ss_tot identically)
        r_ef = residualize(y_c, ef_codes)
        r_efh = residualize(y_c, ef_codes + [host_codes])
        r_blk = residualize(y_c, ef_codes + [host_codes] + ind_codes)
        r_add = residualize(y_c, ef_codes + [host_codes] + ind_codes + [iso_codes])
        r_full = residualize(y_c, ef_codes + [host_codes] + ind_codes + [iso_codes, cell_codes])

        ss_tot = float(np.dot(y_c, y_c))
        ss_ef = max(float(ss_tot - np.dot(r_ef, r_ef)), 0.0)
        ss_h = max(float(np.dot(r_ef, r_ef) - np.dot(r_efh, r_efh)), 0.0)
        ss_ind_in_h = max(float(np.dot(r_efh, r_efh) - np.dot(r_blk, r_blk)), 0.0)
        ss_block = ss_ef + ss_ind_in_h
        ss_i = max(float(np.dot(r_blk, r_blk) - np.dot(r_add, r_add)), 0.0)
        ss_res = float(np.dot(r_full, r_full))
        ss_ih = max(float(np.dot(r_add, r_add) - ss_res), 0.0)

        n_cells = int(cell_codes.max()) + 1
        I_n = int(iso_codes.max()) + 1
        H_n = int(host_codes.max()) + 1
 # Exact non-redundant df for blocks: Flat nests Exp, IndPlant1 nests Host x Exp
        n_flat = int(pd.factorize(sub["flat_full"].astype(str))[0].max()) + 1 if "flat_full" in block_cols else 1
        n_exp = int(pd.factorize(sub["exp"].astype(str))[0].max()) + 1 if "exp" in block_cols else 1
        n_ind = int(ind_codes[0].max()) + 1 if ind_codes else 0
        n_he = int(pd.factorize(sub["host_id"].astype(str) + "-" + sub["exp"].astype(str))[0].max()) + 1 if ind_codes else 0
        df_block = max((n_flat - 1) + max(n_ind - n_he, 0) + max(n_he - H_n - (n_exp - 1), 0), 1)
        df_ih = max(n_cells - I_n - H_n + 1, 1)
        df_res = max(len(sub) - n_cells - df_block, 1)
        ms_leaf = ss_res / df_res
        ms_ih = ss_ih / df_ih
        cell_cnts = np.bincount(cell_codes)
        n_harm = float(len(cell_cnts) / np.sum(1.0 / np.clip(cell_cnts, 1, None)))
        var_ih_raw = (ms_ih - ms_leaf) / max(n_harm, 1e-9)
        f_ih = ms_ih / max(ms_leaf, 1e-12)
        ratio = float(ms_leaf / var_ih_raw) if var_ih_raw > 1e-6 * ms_leaf else np.nan
        s_all = ss_h + ss_i + ss_ih + ss_block + ss_res
        return {
            "n_replicates": int(len(sub)), "n_isolates": I_n, "n_hosts": H_n,
            "host_pct": round(100.0 * ss_h / s_all, 2),
            "isolate_pct": round(100.0 * ss_i / s_all, 2),
            "ih_pct": round(100.0 * ss_ih / s_all, 2),
            "block_pct": round(100.0 * ss_block / s_all, 2),
            "leaf_resid_pct": round(100.0 * ss_res / s_all, 2),
            "f_ih_over_leaf": round(float(f_ih), 3),
            "sigma2_leaf": round(float(ms_leaf), 4),
            "sigma2_ih": round(float(max(var_ih_raw, 0.0)), 4),
            "leaf_to_ih_var_ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
        }

 # Exact weighted unbalanced 2-way ANOVA (never inflates MS_IH under unequal cell sizes)
    y = sub[val_col].to_numpy(dtype=float)
    y_c = y - y.mean()
    cell_codes = pd.factorize(sub["isolate_id"].astype(str) + "||" + sub["host_id"].astype(str))[0]
    iso_codes = pd.factorize(sub["isolate_id"].astype(str))[0]
    host_codes = pd.factorize(sub["host_id"].astype(str))[0]
    I_n, H_n, n_cells = int(iso_codes.max()) + 1, int(host_codes.max()) + 1, int(cell_codes.max()) + 1

 # Exact cell means
    cell_cnts = np.bincount(cell_codes)
    cell_means = np.bincount(cell_codes, weights=y_c) / np.clip(cell_cnts, 1, None)
    r_full = y_c - cell_means[cell_codes]
    ss_res = float(np.dot(r_full, r_full))

 # Exact additive Host + Isolate fit via alternating projections on observation vector (exact weighted LS)
    r_add = y_c.copy()
    for _ in range(30):
        for gc in (iso_codes, host_codes):
            cnts = np.bincount(gc)
            means = np.bincount(gc, weights=r_add) / np.clip(cnts, 1, None)
            r_add -= means[gc]
    ss_ih = max(float(np.dot(r_add, r_add) - ss_res), 0.0)
 # Exact Type-III Host and Isolate conditional SS
    r_h_only = y_c - (np.bincount(host_codes, weights=y_c) / np.bincount(host_codes))[host_codes]
    r_i_only = y_c - (np.bincount(iso_codes, weights=y_c) / np.bincount(iso_codes))[iso_codes]
    ss_tot = float(np.dot(y_c, y_c))
    ss_hi = max(float(ss_tot - np.dot(r_add, r_add)), 0.0)
    ss_i_cond = max(float(np.dot(r_h_only, r_h_only) - np.dot(r_add, r_add)), 0.0)
    ss_h_cond = max(float(np.dot(r_i_only, r_i_only) - np.dot(r_add, r_add)), 0.0)
    sc = ss_hi / max(ss_h_cond + ss_i_cond, 1e-12)
    ss_h, ss_i = ss_h_cond * sc, ss_i_cond * sc

    df_res = max(len(sub) - n_cells, 1)
    df_ih = max(n_cells - I_n - H_n + 1, 1)
    ms_leaf = ss_res / df_res
    ms_ih = ss_ih / df_ih
 # Exact expected coefficient k0 for unbalanced 2-way interaction MS: harmonic mean is conservative & close to exact k0
    n_harm = float(len(cell_cnts) / np.sum(1.0 / np.clip(cell_cnts, 1, None)))
    var_ih_raw = (ms_ih - ms_leaf) / max(n_harm, 1e-9)
    f_ih = ms_ih / max(ms_leaf, 1e-12)
    ratio = float(ms_leaf / var_ih_raw) if var_ih_raw > 1e-6 * ms_leaf else np.nan
    s_all = ss_h + ss_i + ss_ih + ss_res
    return {
        "n_replicates": int(len(sub)), "n_isolates": I_n, "n_hosts": H_n,
        "host_pct": round(100.0 * ss_h / s_all, 2),
        "isolate_pct": round(100.0 * ss_i / s_all, 2),
        "ih_pct": round(100.0 * ss_ih / s_all, 2),
        "block_pct": 0.0,
        "leaf_resid_pct": round(100.0 * ss_res / s_all, 2),
        "f_ih_over_leaf": round(float(f_ih), 3),
        "sigma2_leaf": round(float(ms_leaf), 4),
        "sigma2_ih": round(float(max(var_ih_raw, 0.0)), 4),
        "leaf_to_ih_var_ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
    }


# ---------------------------------------------------------------------------
# Item-Response / Fisher-Information Sentinel Rule
# ---------------------------------------------------------------------------
def select_info_opt(train_mat: np.ndarray, m: int, k: int, bounded_pct: bool) -> np.ndarray:
    """Select m training isolates maximizing Fisher information at the top-k host decision boundary.
    - Bounded [0, 100]% scales: fit additive logit IRT model logit(p_ih) = theta_h + delta_i;
      information of isolate i on boundary host h is I_ih = p_hat_ih * (1 - p_hat_ih).
    - Unbounded multiplicative scales: fit log-additive model log(y_ih) = theta_h + delta_i;
      on raw-scale additive noise, Fisher information for separating boundary hosts scales with
      predicted boundary mean squared / residual variance (i.e., high signal delta_i without extreme I*H leverage).
    """
    n_tr, n_h = train_mat.shape
    if bounded_pct:
        L = logit_bounded(train_mat, eps=0.025)
        mu = float(L.mean())
        theta_h = L.mean(axis=0) - mu
        delta_i = L.mean(axis=1) - mu
        p_hat = expit(mu + delta_i[:, None] + theta_h[None, :])
 # Identify hosts near the top-k boundary (ranks 0 .. min(k+2, n_h-1) on latent theta_h)
        h_ord = np.argsort(theta_h)
        b_hosts = h_ord[: min(k + 3, n_h)]
 # Boundary Fisher information weighted toward the k / k+1 cut (down-weighting saturated/floor cells)
        info_i = np.sum(p_hat[:, b_hosts] * (1.0 - p_hat[:, b_hosts]), axis=1)
 # Mildly penalize extreme I*H outlier residuals so selected sentinels are both informative AND representative
        resid_sd = np.std(L - (mu + delta_i[:, None] + theta_h[None, :]), axis=1)
        score = info_i / (1.0 + 0.25 * resid_sd)
        return np.argsort(score)[-m:]
    else:
 # Unbounded positive scale (B. cinerea lesions or Z. tritici log1p_pycnidia)
        L = log_positive(train_mat, eps=0.05) if np.nanmax(train_mat) > 10.0 else train_mat
        mu = float(L.mean())
        theta_h = L.mean(axis=0) - mu
        delta_i = L.mean(axis=1) - mu
        resid = L - (mu + delta_i[:, None] + theta_h[None, :])
        resid_var = np.var(resid, axis=1) + 1e-4
 # Signal-to-noise discrimination: isolate's empirical host-slope (Finlay-Wilkinson) divided by residual variance
 # Slope b_i = cov(train_mat[i], host_mean) / var(host_mean) -> high slope + low I*H residual = ideal test item
        hm = train_mat.mean(axis=0)
        hm_c = hm - hm.mean()
        var_hm = float(np.dot(hm_c, hm_c)) + 1e-12
        slopes = ((train_mat - train_mat.mean(axis=1, keepdims=True)) @ hm_c) / var_hm
 # Predictability / information = slope^2 / residual_mse (exact linear-model Fisher info)
        pred = train_mat.mean(axis=1, keepdims=True) + slopes[:, None] * hm_c[None, :]
        mse_i = np.mean((train_mat - pred) ** 2, axis=1) + 1e-4
        score = np.where(slopes > 0, (slopes ** 2) / mse_i, -1e9)
        return np.argsort(score)[-m:]


def model_based_host_scores(sub_mat: np.ndarray, bounded_pct: bool, k_mat: np.ndarray | None = None, n_mat: np.ndarray | None = None) -> np.ndarray:
    """Scale-appropriate model-based host scores:
    - Binomial-logit empirical-Bayes MLE for P. sojae (exact k/n)
    - Logit-normal additive estimate (isolate-centered logit mean) for Z. tritici PLACL
    - Log-normal additive estimate (isolate-centered log mean) for B. cinerea & Z. tritici pycnidia.
    """
    if k_mat is not None and n_mat is not None:
 # Regularized binomial logit (adds 0.25 pseudo-count per cell, removes isolate additive offset on logit scale)
        p_sm = (k_mat + 0.25) / (n_mat + 0.5)
        L = np.log(p_sm / (1.0 - p_sm))
        iso_off = L.mean(axis=1, keepdims=True) - L.mean()
        return (L - iso_off).mean(axis=0)
    if bounded_pct:
        L = logit_bounded(sub_mat, eps=0.025)
        iso_off = L.mean(axis=1, keepdims=True) - L.mean()
        return (L - iso_off).mean(axis=0)
    else:
        L = log_positive(sub_mat, eps=0.05)
        iso_off = L.mean(axis=1, keepdims=True) - L.mean()
        return (L - iso_off).mean(axis=0)


# ---------------------------------------------------------------------------
# Genomic Covering & Partial Mantel (Smouse-Long-Sokal 1986)
# ---------------------------------------------------------------------------
def kmedoids_cover(D: np.ndarray, m: int) -> list[int]:
    n = D.shape[0]
    m = min(m, n)
    D_c = np.nan_to_num(D, nan=float(np.nanmedian(D)))
    meds = [int(np.argmin(D_c.mean(axis=1)))]
    if m == 1:
        return meds
    cur_min = D_c[meds[0]].copy()
    for _ in range(1, m):
        cand = np.minimum(cur_min[None, :], D_c).sum(axis=1)
        cand[meds] = np.inf
        nxt = int(np.argmin(cand))
        meds.append(nxt)
        cur_min = np.minimum(cur_min, D_c[nxt])
    for _ in range(25):
        improved = False
        base = float(D_c[meds].min(axis=0).sum())
        for pos in range(m):
            other = [meds[p] for p in range(m) if p != pos]
            omin = D_c[other].min(axis=0)
            cand = np.minimum(omin[None, :], D_c).sum(axis=1)
            cand[other] = np.inf
            bj = int(np.argmin(cand))
            if cand[bj] < base - 1e-9:
                meds[pos] = bj
                base = float(cand[bj])
                improved = True
        if not improved:
            break
    return sorted(meds)


def farthest_point_medoid_init(D: np.ndarray, m: int) -> list[int]:
    D_c = np.nan_to_num(D, nan=float(np.nanmedian(D)))
    n = D_c.shape[0]
    m = min(m, n)
    sel = [int(np.argmin(D_c.mean(axis=1)))]
    min_d = D_c[sel[0]].copy()
    for _ in range(1, m):
        min_d[sel] = -1.0
        nxt = int(np.argmax(min_d))
        sel.append(nxt)
        min_d = np.minimum(min_d, D_c[nxt])
    return sel


def mantel_and_partial(D_gen: np.ndarray, D_phen: np.ndarray, Z_mats: list[np.ndarray] | None, rng: np.random.Generator, n_perm: int = 1000) -> dict[str, float]:
    n = D_gen.shape[0]
    iu = np.triu_indices(n, k=1)
    vec_g, vec_p = D_gen[iu], D_phen[iu]
    ok = ~np.isnan(vec_g) & ~np.isnan(vec_p)
    x0 = np.nan_to_num(D_gen, nan=float(np.nanmean(vec_g[ok])))
    y0 = np.nan_to_num(D_phen, nan=float(np.nanmean(vec_p[ok])))
    vx, vy = x0[iu], y0[iu]
    r_obs = float(pearsonr(vx, vy).statistic)
    rho_obs = fast_spearman(vx, vy)
    vy_c = vy - vy.mean()
    vy_n = np.sqrt(np.dot(vy_c, vy_c))
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        xp = x0[np.ix_(p, p)][iu]
        xp_c = xp - xp.mean()
        rp = float(np.dot(xp_c, vy_c) / (np.sqrt(np.dot(xp_c, xp_c)) * vy_n + 1e-12))
        if rp >= r_obs - 1e-12:
            ge += 1
    out = {"mantel_r": round(r_obs, 4), "mantel_rho": round(rho_obs, 4), "p_perm_le": round((ge + 1.0) / (n_perm + 1.0), 4), "n_isolates": n, "n_pairs": int(ok.sum())}
    if Z_mats:
 # Partial Mantel: residualize vx and vy on Z design matrix (same-field, same-state)
        Z = np.column_stack([np.ones(len(vx))] + [zm[iu] for zm in Z_mats])
        Q, _ = np.linalg.qr(Z)
        rx = vx - Q @ (Q.T @ vx)
        ry = vy - Q @ (Q.T @ vy)
        pr_obs = float(pearsonr(rx, ry).statistic)
        ry_n = np.sqrt(np.dot(ry, ry))
 # Permute rows/cols of residualized genetic matrix R_X (Legendre 2000 residual permutation)
        R_mat = np.zeros((n, n), dtype=float)
        R_mat[iu] = rx
        R_mat = R_mat + R_mat.T
        ge_p = 0
        for _ in range(n_perm):
            p = rng.permutation(n)
            rp_vec = R_mat[np.ix_(p, p)][iu]
            rp_c = rp_vec - rp_vec.mean()
            r_val = float(np.dot(rp_c, ry) / (np.sqrt(np.dot(rp_c, rp_c)) * ry_n + 1e-12))
            if r_val >= pr_obs - 1e-12:
                ge_p += 1
        out["partial_mantel_r"] = round(pr_obs, 4)
        out["partial_p_perm_le"] = round((ge_p + 1.0) / (n_perm + 1.0), 4)
    else:
        out["partial_mantel_r"] = np.nan
        out["partial_p_perm_le"] = np.nan
    return out


# ---------------------------------------------------------------------------
# 1. Variance Decompositions (incl. B. cinerea Blocked Refit) & P. sojae 34-Pair Binomial/Beta-Binomial Audit
# ---------------------------------------------------------------------------
def run_variance_and_repeatability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/6] Variance decompositions (exact weighted + B. cinerea blocked refit) & P. sojae binomial/ICC audit...", flush=True)
    var_rows = []

 # P. sojae
    plong = pd.read_csv(PROC / "psojae_complete_long.csv")
    piso = pd.read_csv(PROC / "psojae_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    rps_hosts = sorted([h for h in plong["host_id"].unique() if not h.startswith("D1_")])
    pisos = piso["isolate_id"].tolist()
    p_pct = plong.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[pisos, rps_hosts].values
    p_bin = plong.pivot(index="isolate_id", columns="host_id", values="binary_from_count").loc[pisos, rps_hosts].values * 100.0
    var_rows.append({"pathosystem": "P. sojae", "subset": "15 Rps differentials (raw % mortality)", "level": "cell_mean_2way", "n_isolates": len(pisos), "n_hosts": len(rps_hosts), "n_obs": p_pct.size, **variance_decomp_2way(p_pct), "block_pct": np.nan, "leaf_resid_pct": np.nan, "f_ih_over_leaf": np.nan, "leaf_to_ih_var_ratio": np.nan})
    var_rows.append({"pathosystem": "P. sojae", "subset": "15 Rps differentials (70% binary call)", "level": "cell_mean_2way", "n_isolates": len(pisos), "n_hosts": len(rps_hosts), "n_obs": p_bin.size, **variance_decomp_2way(p_bin), "block_pct": np.nan, "leaf_resid_pct": np.nan, "f_ih_over_leaf": np.nan, "leaf_to_ih_var_ratio": np.nan})

 # P. sojae 34 replicated isolate pairs: exact Binomial flip expectation, Beta-Binomial MLE overdispersion, & Host-adjusted ICC
    rep34 = pd.read_csv(PROC / "psojae_replicates_34pairs.csv")
    r_rps = rep34[~rep34["host_id"].str.startswith("D1_")].copy()
    r_rps["abs_diff"] = (r_rps["pct_run1"] - r_rps["pct_run2"]).abs()
    r_rps["flip70"] = (r_rps["bin70_run1"] != r_rps["bin70_run2"]).astype(int)
    r_rps["p_pool"] = (r_rps["k_run1"] + r_rps["k_run2"]) / (r_rps["n_run1"] + r_rps["n_run2"])

 # Exact theoretical binomial flip probability per paired cell given pooled p_hat and exact (n1, n2):
 # P(flip) = P(Bin(n1, p)>=0.7*n1)*P(Bin(n2, p)<0.7*n2) + P(Bin(n1, p)<0.7*n1)*P(Bin(n2, p)>=0.7*n2)
    exp_flips = []
    for row in r_rps.itertuples(index=False):
        p = float(row.p_pool)
        t1 = int(np.ceil(0.7 * row.n_run1 - 1e-9))
        t2 = int(np.ceil(0.7 * row.n_run2 - 1e-9))
        p_ge1 = float(binom.sf(t1 - 1, int(row.n_run1), p))
        p_ge2 = float(binom.sf(t2 - 1, int(row.n_run2), p))
        exp_flips.append(p_ge1 * (1.0 - p_ge2) + (1.0 - p_ge1) * p_ge2)
    r_rps["exp_binom_flip"] = exp_flips

    mid_mask = (r_rps["pct_run1"] >= 40.0) & (r_rps["pct_run1"] <= 90.0)
    mid_pool_mask = (r_rps["p_pool"] >= 0.40) & (r_rps["p_pool"] <= 0.90)

 # Fit Beta-Binomial overdispersion parameter rho_bb in (0, 1) via MLE across all 508 paired runs
 # For each pair i, p_pool gives the cell mean; concentration M = (1 - rho_bb) / rho_bb -> alpha = M*p, beta = M*(1-p)
 # Exclude degenerate pooled 0% or 100% cells where both runs agree trivially (or use tiny clip 1e-4)
    p_clip = np.clip(r_rps["p_pool"].values, 0.005, 0.995)
    k1v, n1v = r_rps["k_run1"].values.astype(float), r_rps["n_run1"].values.astype(float)
    k2v, n2v = r_rps["k_run2"].values.astype(float), r_rps["n_run2"].values.astype(float)

    log_comb = (
        gammaln(n1v + 1.0) - gammaln(k1v + 1.0) - gammaln(n1v - k1v + 1.0)
        + gammaln(n2v + 1.0) - gammaln(k2v + 1.0) - gammaln(n2v - k2v + 1.0)
    )

    def neg_ll_bb(log_M: float) -> float:
        M = float(np.exp(log_M))
        a = M * p_clip
        b = M * (1.0 - p_clip)
        ll1 = betaln(k1v + a, n1v - k1v + b) - betaln(a, b)
        ll2 = betaln(k2v + a, n2v - k2v + b) - betaln(a, b)
        return -float(np.sum(ll1 + ll2 + log_comb))

    opt = minimize_scalar(neg_ll_bb, bounds=(np.log(0.2), np.log(1e4)), method="bounded")
    M_hat = float(np.exp(opt.x))
    rho_bb = 1.0 / (M_hat + 1.0)  # intra-run correlation / overdispersion (0 = pure binomial)
 # Exact likelihood ratio vs pure binomial (M -> inf; includes log C(n, k) in both)
    ll_bb = -float(opt.fun)
    ll_bin = float(np.sum(binom.logpmf(k1v, n1v, p_clip) + binom.logpmf(k2v, n2v, p_clip)))
    lrt_stat = max(2.0 * (ll_bb - ll_bin), 0.0)
    lrt_p = float(0.5 * chi2.sf(lrt_stat, 1))
    non_deg = (r_rps["p_pool"].values > 0.0) & (r_rps["p_pool"].values < 1.0)
    p_nd = r_rps["p_pool"].values[non_deg]
    chi2_nd = float(np.sum(
        (k1v[non_deg] - n1v[non_deg] * p_nd) ** 2 / (n1v[non_deg] * p_nd * (1.0 - p_nd))
        + (k2v[non_deg] - n2v[non_deg] * p_nd) ** 2 / (n2v[non_deg] * p_nd * (1.0 - p_nd))
    ))
    phi_pearson = chi2_nd / max(int(non_deg.sum()), 1)

 # Host-adjusted ICC(1,1) (Shrout & Fleiss 1979): remove host means from both runs, compute one-way random ICC across 508 cells
    host_m = r_rps.groupby("host_id")[["pct_run1", "pct_run2"]].transform("mean").mean(axis=1)
    y1_adj = (r_rps["pct_run1"] - host_m).values
    y2_adj = (r_rps["pct_run2"] - host_m).values
    raw_r = float(pearsonr(r_rps["pct_run1"], r_rps["pct_run2"]).statistic)
    host_adj_r = float(pearsonr(y1_adj, y2_adj).statistic)
 # Exact one-way ANOVA ICC(1,1) on host-adjusted paired runs: MSB = 2*var(pair_mean), MSW = mean((y1-y2)^2)/2
    pair_m = 0.5 * (y1_adj + y2_adj)
    msb = 2.0 * float(np.var(pair_m, ddof=1))
    msw = 0.5 * float(np.mean((y1_adj - y2_adj) ** 2))
    icc_host_adj = (msb - msw) / (msb + msw)
 # Unadjusted ICC(1,1)
    pm_raw = 0.5 * (r_rps["pct_run1"].values + r_rps["pct_run2"].values)
    msb_r = 2.0 * float(np.var(pm_raw, ddof=1))
    msw_r = 0.5 * float(np.mean((r_rps["pct_run1"].values - r_rps["pct_run2"].values) ** 2))
    icc_raw = (msb_r - msw_r) / (msb_r + msw_r)

 # 4-way on 34 replicated pairs
    r_long = pd.concat([
        r_rps[["isolate_id", "host_id", "pct_run1"]].rename(columns={"pct_run1": "pct"}),
        r_rps[["isolate_id", "host_id", "pct_run2"]].rename(columns={"pct_run2": "pct"}),
    ], ignore_index=True)
    v4_p34 = variance_decomp_4way_exact(r_long, "pct")
    var_rows.append({"pathosystem": "P. sojae", "subset": "34 replicated isolates (2 runs x 15 Rps)", "level": "raw_replicate_4way", "n_isolates": v4_p34["n_isolates"], "n_hosts": v4_p34["n_hosts"], "n_obs": v4_p34["n_replicates"], "host_pct": v4_p34["host_pct"], "isolate_pct": v4_p34["isolate_pct"], "ih_pct": v4_p34["ih_pct"], "ih_to_h": round(v4_p34["ih_pct"] / max(v4_p34["host_pct"], 1e-9), 3), "block_pct": 0.0, "leaf_resid_pct": v4_p34["leaf_resid_pct"], "f_ih_over_leaf": v4_p34["f_ih_over_leaf"], "leaf_to_ih_var_ratio": v4_p34["leaf_to_ih_var_ratio"]})

    big_disc = r_rps[r_rps["abs_diff"] >= 60.0][["isolate_id", "state", "host_id", "k_run1", "n_run1", "pct_run1", "k_run2", "n_run2", "pct_run2", "abs_diff"]].sort_values("abs_diff", ascending=False)
    big_disc.to_csv(RES / "psojae_extreme_discordant_pairs.csv", index=False)

    d1_rep = rep34[rep34["line_code"] == "D1"]
    rep_audit = pd.DataFrame([{
        "n_replicated_isolates": int(rep34["isolate_id"].nunique()),
        "n_paired_rps_cells": int(len(r_rps)),
        "d1_run1_below70_count": int((d1_rep["pct_run1"] < 70.0).sum()),
        "raw_pearson_r": round(raw_r, 4),
        "icc_unadjusted": round(icc_raw, 4),
        "host_adjusted_r": round(host_adj_r, 4),
        "icc_host_adjusted": round(icc_host_adj, 4),
        "mad_pct": round(float(r_rps["abs_diff"].mean()), 2),
        "obs_flip70_overall_pct": round(100.0 * float(r_rps["flip70"].mean()), 2),
        "exp_binom_flip70_overall_pct": round(100.0 * float(r_rps["exp_binom_flip"].mean()), 2),
        "obs_flip70_40_90_run1_pct": round(100.0 * float(r_rps.loc[mid_mask, "flip70"].mean()), 2),
        "obs_flip70_40_90_pooled_pct": round(100.0 * float(r_rps.loc[mid_pool_mask, "flip70"].mean()), 2),
        "exp_binom_flip70_40_90_pooled_pct": round(100.0 * float(r_rps.loc[mid_pool_mask, "exp_binom_flip"].mean()), 2),
        "beta_binom_M_hat": round(M_hat, 3),
        "beta_binom_overdispersion_rho": round(rho_bb, 4),
        "lrt_2dll_vs_binomial": round(lrt_stat, 2),
        "lrt_p_value": float(f"{lrt_p:.3e}"),
        "pearson_chi2_nondeg": round(chi2_nd, 2),
        "pearson_df_nondeg": int(non_deg.sum()),
        "pearson_dispersion_phi": round(phi_pearson, 3),
        "extreme_ge60_pairs": int(len(big_disc)),
        "extreme_ge60_pct": round(100.0 * len(big_disc) / len(r_rps), 2),
        "same_octal_isolates": int(rep34.groupby("isolate_id")["same_octal_sheet"].first().sum()),
    }])
    rep_audit.to_csv(RES / "psojae_repeatability_audit.csv", index=False)

 # Z. tritici (11,019 greenhouse seedling leaves)
    zcell = pd.read_csv(PROC / "zymo_cell_means.csv")
    zrep = pd.read_csv(PROC / "zymo_replicates.csv")
    ziso = pd.read_csv(PROC / "zymo_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    zhosts = sorted(zcell["host_id"].unique().tolist())
    zisos = ziso["isolate_id"].tolist()
    for label, ccol, rcol in [
        ("PLACL (% leaf necrosis)", "placl_mean", "placl"),
        ("log1p(Pycnidia/cm2)", "log1p_pycnidia_mean", "log1p_pycnidia"),
        ("Pycnidia/cm2 (raw)", "pycnidia_mean", "pycnidia_cm2"),
    ]:
        M = zcell.pivot(index="isolate_id", columns="host_id", values=ccol).loc[zisos, zhosts].values
        var_rows.append({"pathosystem": "Z. tritici", "subset": label, "level": "cell_mean_2way", "n_isolates": len(zisos), "n_hosts": len(zhosts), "n_obs": M.size, **variance_decomp_2way(M), "block_pct": np.nan, "leaf_resid_pct": np.nan, "f_ih_over_leaf": np.nan, "leaf_to_ih_var_ratio": np.nan})
        var4 = variance_decomp_4way_exact(zrep, rcol)
        var_rows.append({"pathosystem": "Z. tritici", "subset": label, "level": "raw_replicate_4way", "n_isolates": var4["n_isolates"], "n_hosts": var4["n_hosts"], "n_obs": var4["n_replicates"], "host_pct": var4["host_pct"], "isolate_pct": var4["isolate_pct"], "ih_pct": var4["ih_pct"], "ih_to_h": round(var4["ih_pct"] / max(var4["host_pct"], 1e-9), 3), "block_pct": 0.0, "leaf_resid_pct": var4["leaf_resid_pct"], "f_ih_over_leaf": var4["f_ih_over_leaf"], "leaf_to_ih_var_ratio": var4["leaf_to_ih_var_ratio"]})

 # Export Z. tritici 12-cultivar profile summary directly (required for clean-folder reproduction of Fig. 5C)
    zcult = (
        zcell.groupby("host_id", as_index=False)
        .agg(
            mean_placl=("placl_mean", "mean"),
            sd_placl=("placl_mean", "std"),
            mean_pycnidia=("pycnidia_mean", "mean"),
            mean_log1p_pycnidia=("log1p_pycnidia_mean", "mean"),
        )
    )
    zcult["rank_placl"] = zcult["mean_placl"].rank(method="min").astype(int)
    zcult["rank_pycnidia"] = zcult["mean_pycnidia"].rank(method="min").astype(int)
    zcult["rank_log1p_pycnidia"] = zcult["mean_log1p_pycnidia"].rank(method="min").astype(int)
    zcult["rank_gap_placl_minus_pyc"] = zcult["rank_placl"] - zcult["rank_pycnidia"]
    zcult = zcult.sort_values("mean_placl").reset_index(drop=True)
    zcult.to_csv(RES / "zymo_cultivar_profiles.csv", index=False)

 # B. cinerea (sequential orthogonal sums-of-squares decomposition matching Caseys et al. block factors)
    bls = pd.read_csv(PROC / "botrytis_lsmeans_long.csv")
    brep = pd.read_csv(PROC / "botrytis_replicates_7sp.csv").dropna(subset=["lesion_cm2"]).copy()
    brep["flat_full"] = brep["exp"].astype(str) + "-" + brep["rep"].astype(str) + "-" + brep["flat"].astype(str)
    brep["ind_plant1"] = brep["host_id"].astype(str) + "-" + brep["exp"].astype(str) + "-" + brep["ind_plant"].astype(str)
    crop_sp = sorted([s for s in bls["crop_species"].unique() if s != "Athaliana"])

 # Compute exact empirical cell-level within-cell mean-SD power-law scaling (log SD ~ alpha * log mean) per crop
 # across all 7 crop species (cells with >= 3 leaves, mean > 0.01, SD > 0.001) and export to botrytis_noise_scaling.csv
    noise_rows = []
    for sp in crop_sp:
        g_sp = brep[brep["crop_species"] == sp]
        cs = g_sp.groupby(["isolate_id", "host_id"])["lesion_cm2"].agg(["count", "mean", "std"]).dropna()
        cs = cs[(cs["count"] >= 3) & (cs["mean"] > 0.01) & (cs["std"] > 0.001)]
        lx, ly = np.log(cs["mean"].values), np.log(cs["std"].values)
        slope_cell, intercept_cell = np.polyfit(lx, ly, 1)
        r_cell = float(pearsonr(lx, ly).statistic)
        iso_s = cs.groupby("isolate_id").agg(m=("mean", "mean"), sd=("std", "mean"))
        ok_i = (iso_s["m"] > 0.01) & (iso_s["sd"] > 0.001)
        slope_iso = float(np.polyfit(np.log(iso_s.loc[ok_i, "m"]), np.log(iso_s.loc[ok_i, "sd"]), 1)[0])
        noise_rows.append({
            "crop_species": sp,
            "n_cells_ge3": int(len(cs)),
            "cell_alpha_slope": round(float(slope_cell), 4),
            "cell_intercept": round(float(intercept_cell), 4),
            "cell_loglog_r": round(r_cell, 4),
            "isolate_alpha_slope": round(slope_iso, 4),
        })
    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(RES / "botrytis_noise_scaling.csv", index=False)
    print(f"  B. cinerea empirical cell mean-SD alpha (7 crops): median={df_noise['cell_alpha_slope'].median():.3f}, range={df_noise['cell_alpha_slope'].min():.3f}-{df_noise['cell_alpha_slope'].max():.3f}")
    b2_list, b4_list = [], []
    for sp in crop_sp:
        sub = bls[bls["crop_species"] == sp]
        piv = sub.pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").dropna(axis=0, how="any")
        vd = variance_decomp_2way(piv.values)
        b2_list.append(vd)
        var_rows.append({"pathosystem": "B. cinerea", "subset": f"{sp} (12 genotypes)", "level": "cell_mean_2way", "n_isolates": piv.shape[0], "n_hosts": piv.shape[1], "n_obs": piv.size, **vd, "block_pct": np.nan, "leaf_resid_pct": np.nan, "f_ih_over_leaf": np.nan, "leaf_to_ih_var_ratio": np.nan})

        rsub = brep[brep["crop_species"] == sp]
        var4 = variance_decomp_4way_exact(rsub, "lesion_cm2", block_cols=["exp", "flat_full", "ind_plant1"])
        b4_list.append(var4)
        var_rows.append({"pathosystem": "B. cinerea", "subset": f"{sp} (blocked mixed-model refit)", "level": "raw_replicate_blocked", "n_isolates": var4["n_isolates"], "n_hosts": var4["n_hosts"], "n_obs": var4["n_replicates"], "host_pct": var4["host_pct"], "isolate_pct": var4["isolate_pct"], "ih_pct": var4["ih_pct"], "ih_to_h": round(var4["ih_pct"] / max(var4["host_pct"], 1e-9), 3), "block_pct": var4["block_pct"], "leaf_resid_pct": var4["leaf_resid_pct"], "f_ih_over_leaf": var4["f_ih_over_leaf"], "leaf_to_ih_var_ratio": var4["leaf_to_ih_var_ratio"]})

 # 7-crop mean summary rows
    var_rows.append({
        "pathosystem": "B. cinerea", "subset": "7-crop mean (LSmeans)", "level": "cell_mean_2way",
        "n_isolates": 98, "n_hosts": 84,
        "n_obs": sum(r["n_obs"] for r in var_rows if r["pathosystem"] == "B. cinerea" and r["level"] == "cell_mean_2way"),
        "host_pct": round(float(np.mean([d["host_pct"] for d in b2_list])), 2),
        "isolate_pct": round(float(np.mean([d["isolate_pct"] for d in b2_list])), 2),
        "ih_pct": round(float(np.mean([d["ih_pct"] for d in b2_list])), 2),
        "ih_to_h": round(float(np.mean([d["ih_to_h"] for d in b2_list])), 3),
        "block_pct": np.nan, "leaf_resid_pct": np.nan, "f_ih_over_leaf": np.nan, "leaf_to_ih_var_ratio": np.nan,
    })
    pos_b = [d["leaf_to_ih_var_ratio"] for d in b4_list if not pd.isna(d["leaf_to_ih_var_ratio"])]
    var_rows.append({
        "pathosystem": "B. cinerea", "subset": "7-crop mean (42,200 leaves, blocked)", "level": "raw_replicate_blocked",
        "n_isolates": 98, "n_hosts": 84,
        "n_obs": int(sum(d["n_replicates"] for d in b4_list)),
        "host_pct": round(float(np.mean([d["host_pct"] for d in b4_list])), 2),
        "isolate_pct": round(float(np.mean([d["isolate_pct"] for d in b4_list])), 2),
        "ih_pct": round(float(np.mean([d["ih_pct"] for d in b4_list])), 2),
        "ih_to_h": round(float(np.mean([d["ih_pct"] / max(d["host_pct"], 1e-9) for d in b4_list])), 3),
        "block_pct": round(float(np.mean([d["block_pct"] for d in b4_list])), 2),
        "leaf_resid_pct": round(float(np.mean([d["leaf_resid_pct"] for d in b4_list])), 2),
        "f_ih_over_leaf": round(float(np.mean([d["f_ih_over_leaf"] for d in b4_list])), 3),
        "leaf_to_ih_var_ratio": round(float(np.median(pos_b)), 3) if pos_b else np.nan,
    })

    df_var = pd.DataFrame(var_rows)
    df_var.to_csv(RES / "variance_decomposition.csv", index=False)
    print(f"  B. cinerea blocked refit 7-crop mean: F_IH={df_var.iloc[-1]['f_ih_over_leaf']}, block%={df_var.iloc[-1]['block_pct']}%, leaf_resid%={df_var.iloc[-1]['leaf_resid_pct']}%, median ratio={df_var.iloc[-1]['leaf_to_ih_var_ratio']} (n_pos={len(pos_b)}/7)")
    print(f"  P. sojae 34 pairs: obs flip={rep_audit['obs_flip70_overall_pct'].iloc[0]}% vs exp binom={rep_audit['exp_binom_flip70_overall_pct'].iloc[0]}%; 40-90% obs={rep_audit['obs_flip70_40_90_pooled_pct'].iloc[0]}% vs exp={rep_audit['exp_binom_flip70_40_90_pooled_pct'].iloc[0]}%; BB rho={rho_bb:.4f}; host-adj ICC={icc_host_adj:.3f}")
    return df_var, rep_audit, big_disc


# ---------------------------------------------------------------------------
# 2. Core Evaluation Engine: Shared Panel Draws, Scale Sensitivity, InfoOpt-IRT,
# Model-Based Ranking, Split-Host Proxy, 2x2 Threshold Scoring,
# Equal-m Spatial Sampling, and Z. tritici 4-Region Joint-Oregon Cross-Endpoint Transfer
# ---------------------------------------------------------------------------
def build_shared_crn_bank(rng: np.random.Generator, pool_indices: np.ndarray, m_list: list[int], n_draws: int, n_hosts: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pre-generate exact shared random panels and tie-breaking jitters per (fold, m)."""
    bank = {}
    for m in m_list:
        if len(pool_indices) < m:
            continue
        panels = np.empty((n_draws, m), dtype=int)
        for d in range(n_draws):
            panels[d] = rng.choice(pool_indices, size=m, replace=False)
        jitters = rng.uniform(0.0, 1e-9, size=(n_draws, n_hosts))
        bank[m] = (panels, jitters)
    return bank


def run_unified_benchmarks(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    print("[2/6] Running unified CRN benchmarks (P. sojae, Z. tritici 4-region & 5-pop, B. cinerea)...", flush=True)
    fold_rows = []   # Universal fold-level table (pathosystem x trait x scale x split x fold x m x k x strategy)
    dec_rows = []    # Deciles WITH isolate-bootstrap uncertainty bands
    thresh_2x2_rows = []
    split_host_rows = []
    spatial_rows = []
    ce_rows = []

 # =========================================================================
 # A. PHYTOPHTHORA SOJAE (455 isolates, 3 US states IN/KY/OH, 15 Rps lines)
 # =========================================================================
    plong = pd.read_csv(PROC / "psojae_complete_long.csv")
    pfirst = pd.read_csv(PROC / "psojae_first_run_complete_long.csv")
    piso = pd.read_csv(PROC / "psojae_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    rps_hosts = sorted([h for h in plong["host_id"].unique() if not h.startswith("D1_")])
    pisos = piso["isolate_id"].tolist()
    pstates = piso["state"].values
    plocs = (piso["state"].astype(str) + "::" + piso["location_id"].astype(str)).values

    p_pct = plong.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[pisos, rps_hosts].values
    p_k = plong.pivot(index="isolate_id", columns="host_id", values="k_susc").loc[pisos, rps_hosts].values
    p_n = plong.pivot(index="isolate_id", columns="host_id", values="n_plants").loc[pisos, rps_hosts].values
    p_bin = (p_pct >= 70.0 - 1e-9).astype(float) * 100.0
    p_sheet = plong.pivot(index="isolate_id", columns="host_id", values="binary_70").loc[pisos, rps_hosts].values * 100.0
    p_first = pfirst.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[pisos, rps_hosts].values
    p_logit = logit_bounded(p_pct, eps=0.025)

 # Deciles with exact fold-matched min-max bounds for out-of-state rho and bootstrap bands for relative host SD
    p_base_sd = float(np.std(p_pct.mean(axis=0), ddof=1))
    p_qb = pd.qcut(p_pct.mean(axis=1), 10, labels=False, duplicates="drop")
    for b in sorted(np.unique(p_qb)):
        ib = np.where(p_qb == b)[0]
        sub = p_pct[ib]
        b_sds = [float(np.std(p_pct[rng.choice(ib, size=len(ib), replace=True)].mean(axis=0), ddof=1) / p_base_sd) for _ in range(200)]
        o_rhos = [fast_spearman(p_pct[ib[pstates[ib] != st]].mean(axis=0), p_pct[pstates == st].mean(axis=0)) for st in sorted(np.unique(pstates))]
        dec_rows.append({
            "pathosystem": "P. sojae", "trait": "pct_susc", "decile": int(b + 1), "n_isolates": int(len(ib)),
            "mean_virulence": round(float(sub.mean()), 2),
            "rel_host_sd": round(float(np.std(sub.mean(axis=0), ddof=1) / p_base_sd), 4),
            "rel_host_sd_lo": round(float(np.percentile(b_sds, 2.5)), 4), "rel_host_sd_hi": round(float(np.percentile(b_sds, 97.5)), 4),
            "ceil90_pct": round(float(100.0 * np.mean(sub >= 90.0)), 2),
            "out_of_pop_rho": round(float(np.mean(o_rhos)), 4),
            "rho_lo": round(float(np.min(o_rhos)), 4), "rho_hi": round(float(np.max(o_rhos)), 4),
        })

 # Shared CRN draw bank for P. sojae LOSO (3 states)
    n_mc_p = int(CFG["n_mc_psojae"])
    p_unique_st = sorted(np.unique(pstates).tolist())
    p_crn = {}
    for st in p_unique_st:
        tr = np.where(pstates != st)[0]
        p_crn[st] = build_shared_crn_bank(rng, tr, M_GRID, n_mc_p, len(rps_hosts))

 # Evaluate P. sojae across BOTH scales (Raw_Pct & Logit_Latent) & k in {1, 2, 3}
    for st in p_unique_st:
        te = np.where(pstates == st)[0]
        tr = np.where(pstates != st)[0]
        truth_raw = p_pct[te].mean(axis=0)
        truth_bin = p_bin[te].mean(axis=0)
        truth_logit = p_logit[te].mean(axis=0)

 # Virulence pools strictly inside training states (both raw and logit scale)
        v_raw = p_pct[tr].mean(axis=1)
        q25, q75 = np.percentile(v_raw, [25, 75])
        pool_q1, pool_mid, pool_q4 = tr[v_raw <= q25], tr[(v_raw > q25) & (v_raw < q75)], tr[v_raw >= q75]

        v_log = p_logit[tr].mean(axis=1)
        lq25, lq75 = np.percentile(v_log, [25, 75])
        lpool_q1, lpool_mid, lpool_q4 = tr[v_log <= lq25], tr[(v_log > lq25) & (v_log < lq75)], tr[v_log >= lq75]

 # Pre-draw quartile CRN panels per m
        q_banks = {
            "Q1_Low": build_shared_crn_bank(rng, pool_q1, M_VIR, n_mc_p, len(rps_hosts)),
            "Q2Q3_Mid": build_shared_crn_bank(rng, pool_mid, M_VIR, n_mc_p, len(rps_hosts)),
            "Q4_High": build_shared_crn_bank(rng, pool_q4, M_VIR, n_mc_p, len(rps_hosts)),
            "LQ1_Low": build_shared_crn_bank(rng, lpool_q1, M_VIR, n_mc_p, len(rps_hosts)),
            "LQ2Q3_Mid": build_shared_crn_bank(rng, lpool_mid, M_VIR, n_mc_p, len(rps_hosts)),
            "LQ4_High": build_shared_crn_bank(rng, lpool_q4, M_VIR, n_mc_p, len(rps_hosts)),
        }

 # Pre-compute MultiState_Strat CRN panels & strict equal-m SingleField pools (len >= m ONLY)
        tr_st = pstates[tr]
        tr_loc = plocs[tr]
        st_list = sorted(np.unique(tr_st).tolist())
        loc_counts = pd.Series(tr_loc).value_counts()

        for m in M_GRID + [999]:
            m_lbl = "AllTrain" if m == 999 else str(m)
 # 1) 2x2 Threshold x Truth matrix (k=2 primary + k=1,3 at m=3,5) + Intermediate-Rps subset
            if m == 999:
                panels = tr[None, :]
                jits = np.zeros((1, len(rps_hosts)))
            else:
                panels, jits = p_crn[st][m]

            scoring_defs = [
                ("Raw_pct", p_pct, False),
                ("FirstRun_pct", p_first, False),
                ("Binary_70", p_bin, False),
                ("Binary_Sheet", p_sheet, False),
                ("Binary_50", (p_pct >= 50.0 - 1e-9).astype(float) * 100.0, False),
                ("Logit_Model", None, True),
            ]
            for sc_name, smat, is_mod in scoring_defs:
                for tr_name, tr_vec in [("Truth_MeanMort", truth_raw), ("Truth_PropVir70", truth_bin)]:
                    regs, nregs, rhos, sms = [], [], [], []
                    for d_idx in range(len(panels)):
                        pk = panels[d_idx]
                        sc = model_based_host_scores(p_pct[pk], True, p_k[pk], p_n[pk]) if is_mod else smat[pk].mean(axis=0)
                        ev = eval_selection(sc, tr_vec, 2, jits[d_idx])
                        regs.append(ev["regret"]); nregs.append(ev["norm_regret"]); rhos.append(ev["rho"]); sms.append(ev["severe_miss"])
                    lo, hi = mc_precision_ci(np.array(regs), rng, 300)
                    thresh_2x2_rows.append({
                        "fold": st, "m": m_lbl, "m_num": m, "scoring": sc_name, "truth_def": tr_name, "subset": "All15_Rps",
                        "regret": round(float(np.mean(regs)), 4), "mc_ci_lo": round(lo, 4), "mc_ci_hi": round(hi, 4),
                        "norm_regret": round(float(np.mean(nregs)), 4), "rho": round(float(np.mean(rhos)), 4), "severe_miss": round(float(np.mean(sms)), 4),
                    })
 # Retained intermediate-efficacy Rps subset defined within training states only
                inter_gene_idx = np.argsort(p_pct[tr].mean(axis=0))[2:]
                if sc_name in ("Raw_pct", "Binary_70", "Logit_Model") and m in (3, 5, 10, 999):
                    for tr_name, tr_vec in [("Truth_MeanMort", truth_raw[inter_gene_idx]), ("Truth_PropVir70", truth_bin[inter_gene_idx])]:
                        regs, rhos = [], []
                        for d_idx in range(len(panels)):
                            pk = panels[d_idx]
                            sc_full = model_based_host_scores(p_pct[pk], True, p_k[pk], p_n[pk]) if is_mod else smat[pk].mean(axis=0)
                            ev = eval_selection(sc_full[inter_gene_idx], tr_vec, 2, jits[d_idx][inter_gene_idx])
                            regs.append(ev["regret"]); rhos.append(ev["rho"])
                        thresh_2x2_rows.append({
                            "fold": st, "m": m_lbl, "m_num": m, "scoring": sc_name, "truth_def": tr_name, "subset": f"InterEff_{len(inter_gene_idx)}Rps",
                            "regret": round(float(np.mean(regs)), 4), "mc_ci_lo": np.nan, "mc_ci_hi": np.nan,
                            "norm_regret": np.nan, "rho": round(float(np.mean(rhos)), 4), "severe_miss": np.nan,
                        })

 # 2) Strict Equal-m Spatial comparison (SingleField ONLY when n_field >= m and >= 3 eligible fields)
            if m != 999:
                elig_locs = loc_counts[loc_counts >= m].index.tolist()
 # Multi-state stratified draws
                ms_regs, ms_nregs, ms_rhos, ms_sms = [], [], [], []
                for d_idx in range(n_mc_p):
                    perm_s = rng.permutation(st_list)
                    pools = {ss: rng.permutation(tr[tr_st == ss]).tolist() for ss in perm_s}
                    pk_ms, si = [], 0
                    while len(pk_ms) < m:
                        ss = perm_s[si % len(perm_s)]
                        if pools[ss]:
                            pk_ms.append(pools[ss].pop())
                        si += 1
                    ev = eval_selection(p_pct[pk_ms].mean(axis=0), truth_raw, 2, jits[d_idx])
                    ms_regs.append(ev["regret"]); ms_nregs.append(ev["norm_regret"]); ms_rhos.append(ev["rho"]); ms_sms.append(ev["severe_miss"])
 # Random CRN (identical to thresh_2x2 Raw_pct)
                r_regs, r_nregs, r_rhos, r_sms = [], [], [], []
                for d_idx in range(n_mc_p):
                    ev = eval_selection(p_pct[panels[d_idx]].mean(axis=0), truth_raw, 2, jits[d_idx])
                    r_regs.append(ev["regret"]); r_nregs.append(ev["norm_regret"]); r_rhos.append(ev["rho"]); r_sms.append(ev["severe_miss"])

                spatial_rows.append({"pathosystem": "P. sojae", "trait": "pct_susc", "split": "LOSO_3states", "fold": st, "m": m, "realized_m": m, "n_elig_fields": int(len(loc_counts)), "strategy": "Random", "regret": round(float(np.mean(r_regs)), 4), "norm_regret": round(float(np.mean(r_nregs)), 4), "rho": round(float(np.mean(r_rhos)), 4), "severe_miss": round(float(np.mean(r_sms)), 4)})
                spatial_rows.append({"pathosystem": "P. sojae", "trait": "pct_susc", "split": "LOSO_3states", "fold": st, "m": m, "realized_m": m, "n_elig_fields": int(len(loc_counts)), "strategy": "MultiState_Strat", "regret": round(float(np.mean(ms_regs)), 4), "norm_regret": round(float(np.mean(ms_nregs)), 4), "rho": round(float(np.mean(ms_rhos)), 4), "severe_miss": round(float(np.mean(ms_sms)), 4)})

                if len(elig_locs) >= 3:
                    sf_regs, sf_nregs, sf_rhos, sf_sms = [], [], [], []
                    for d_idx in range(n_mc_p):
                        fl = elig_locs[int(rng.integers(0, len(elig_locs)))]
                        fl_pool = tr[tr_loc == fl]
                        assert len(fl_pool) >= m, "Realized m check failed"
                        pk_sf = rng.choice(fl_pool, size=m, replace=False)
                        ev = eval_selection(p_pct[pk_sf].mean(axis=0), truth_raw, 2, jits[d_idx])
                        sf_regs.append(ev["regret"]); sf_nregs.append(ev["norm_regret"]); sf_rhos.append(ev["rho"]); sf_sms.append(ev["severe_miss"])
                    spatial_rows.append({"pathosystem": "P. sojae", "trait": "pct_susc", "split": "LOSO_3states", "fold": st, "m": m, "realized_m": m, "n_elig_fields": int(len(elig_locs)), "strategy": "SingleField_ExactM", "regret": round(float(np.mean(sf_regs)), 4), "norm_regret": round(float(np.mean(sf_nregs)), 4), "rho": round(float(np.mean(sf_rhos)), 4), "severe_miss": round(float(np.mean(sf_sms)), 4)})
                else:
                    spatial_rows.append({"pathosystem": "P. sojae", "trait": "pct_susc", "split": "LOSO_3states", "fold": st, "m": m, "realized_m": np.nan, "n_elig_fields": int(len(elig_locs)), "strategy": "SingleField_ExactM", "regret": np.nan, "norm_regret": np.nan, "rho": np.nan, "severe_miss": np.nan})

 # 3) Virulence strategies, scale sensitivity, and model-based ranking for m in M_VIR and k in {1, 2, 3}
            if m in M_VIR:
                k_list = [1, 2, 3] if m in (3, 5) else [2]
                for k_val in k_list:
 # Raw scale strategies (InfoOpt_IRT is evaluated under Two-Way Host x Pathogen Holdout)
                    idx_ext = tr[np.argsort(v_raw)[-m:]]
                    for sname, s_panels, s_jits, use_mod in [
                        ("Random", panels, jits, False),
                        ("Random_LogitModel", panels, jits, True),
                        ("Q1_Low", q_banks["Q1_Low"][m][0], q_banks["Q1_Low"][m][1], False),
                        ("Q2Q3_Mid", q_banks["Q2Q3_Mid"][m][0], q_banks["Q2Q3_Mid"][m][1], False),
                        ("Q4_High", q_banks["Q4_High"][m][0], q_banks["Q4_High"][m][1], False),
                        ("ExtremeTopM", idx_ext[None, :], np.zeros((1, len(rps_hosts))), False),
                    ]:
                        rs, nrs, rhs, sms = [], [], [], []
                        for di in range(len(s_panels)):
                            pk = s_panels[di]
                            sc = model_based_host_scores(p_pct[pk], True, p_k[pk], p_n[pk]) if use_mod else p_pct[pk].mean(axis=0)
                            ev = eval_selection(sc, truth_raw, k_val, s_jits[di])
                            rs.append(ev["regret"]); nrs.append(ev["norm_regret"]); rhs.append(ev["rho"]); sms.append(ev["severe_miss"])
                        fold_rows.append({"pathosystem": "P. sojae", "trait": "pct_susc", "scale": "raw", "split": "LOSO_3states", "fold": st, "m": m, "k": k_val, "strategy": sname, "regret": round(float(np.mean(rs)), 4), "norm_regret": round(float(np.mean(nrs)), 4), "rho": round(float(np.mean(rhs)), 4), "severe_miss": round(float(np.mean(sms)), 4)})

 # Variance-stabilized logit scale (both training scores and target truth on logit scale)
                    if k_val == 2:
                        idx_lext = tr[np.argsort(v_log)[-m:]]
                        for sname, s_panels, s_jits in [
                            ("Random", panels, jits),
                            ("Q1_Low", q_banks["LQ1_Low"][m][0], q_banks["LQ1_Low"][m][1]),
                            ("Q2Q3_Mid", q_banks["LQ2Q3_Mid"][m][0], q_banks["LQ2Q3_Mid"][m][1]),
                            ("Q4_High", q_banks["LQ4_High"][m][0], q_banks["LQ4_High"][m][1]),
                            ("ExtremeTopM", idx_lext[None, :], np.zeros((1, len(rps_hosts)))),
                        ]:
                            rs, nrs, rhs, sms = [], [], [], []
                            for di in range(len(s_panels)):
                                pk = s_panels[di]
                                ev = eval_selection(p_logit[pk].mean(axis=0), truth_logit, 2, s_jits[di])
                                rs.append(ev["regret"]); nrs.append(ev["norm_regret"]); rhs.append(ev["rho"]); sms.append(ev["severe_miss"])
                            fold_rows.append({"pathosystem": "P. sojae", "trait": "logit_mortality", "scale": "transformed", "split": "LOSO_3states", "fold": st, "m": m, "k": 2, "strategy": sname, "regret": round(float(np.mean(rs)), 4), "norm_regret": round(float(np.mean(nrs)), 4), "rho": round(float(np.mean(rhs)), 4), "severe_miss": round(float(np.mean(sms)), 4)})

 # Split-host virulence proxy for P. sojae (rank isolates on 7 Rps lines, evaluate on remaining 8 lines, 50 host splits x 3 states)
    n_sh = int(CFG["n_split_host_psojae"])
    for m in [1, 3, 5]:
        sh_acc = {s: [] for s in ["Random", "Q1_Proxy7", "Q2Q3_Proxy7", "Q4_Proxy7", "ExtremeTopM_Proxy7"]}
        for sp_i in range(n_sh):
            hperm = rng.permutation(len(rps_hosts))
            h_proxy, h_eval = hperm[:7], hperm[7:]
            for st in p_unique_st:
                te = np.where(pstates == st)[0]
                tr = np.where(pstates != st)[0]
                truth_e = p_pct[te][:, h_eval].mean(axis=0)
                vp = p_pct[tr][:, h_proxy].mean(axis=1)
                q25, q75 = np.percentile(vp, [25, 75])
                jit = rng.uniform(0.0, 1e-9, size=len(h_eval))
                for sname, pool in [("Random", tr), ("Q1_Proxy7", tr[vp <= q25]), ("Q2Q3_Proxy7", tr[(vp > q25) & (vp < q75)]), ("Q4_Proxy7", tr[vp >= q75])]:
 # 10 draws per host-split x state -> 1,500 draws total
                    rr = [eval_selection(p_pct[rng.choice(pool, size=m, replace=False)][:, h_eval].mean(axis=0), truth_e, 2, jit)["regret"] for _ in range(10)]
                    sh_acc[sname].append(float(np.mean(rr)))
                p_ext = tr[np.argsort(vp)[-m:]]
                sh_acc["ExtremeTopM_Proxy7"].append(eval_selection(p_pct[p_ext][:, h_eval].mean(axis=0), truth_e, 2, jit)["regret"])
        for sname, vals in sh_acc.items():
            lo, hi = mc_precision_ci(np.array(vals), rng, 300)
            split_host_rows.append({"m": m, "strategy": sname, "n_host_splits": n_sh, "regret_pct": round(float(np.mean(vals)), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})

 # Also Illinois 14 binary-only isolates using shared CRN panels across all three scoring rules
    il_df = pd.read_csv(PROC / "psojae_il_binary_only.csv")
    il_piv = il_df[~il_df["host_id"].str.startswith("D1_")].pivot(index="isolate_id", columns="host_id", values="binary_sheet").reindex(columns=rps_hosts)
    valid_il = [c for c in rps_hosts if il_piv[c].notna().sum() >= 10]
    il_truth = il_piv[valid_il].mean(axis=0).values * 100.0
    il_idx = [rps_hosts.index(c) for c in valid_il]
    il_crn = build_shared_crn_bank(rng, np.arange(len(pisos)), [1, 2, 3, 5, 10, 20], n_mc_p, len(valid_il))
    for m in [1, 2, 3, 5, 10, 20, 999]:
        m_lbl = "AllTrain" if m == 999 else str(m)
        panels_il, jits_il = (np.arange(len(pisos))[None, :], np.zeros((1, len(valid_il)))) if m == 999 else il_crn[m]
        for sc_name, smat in [("Raw_pct", p_pct[:, il_idx]), ("Binary_70", p_bin[:, il_idx]), ("Binary_Sheet", p_sheet[:, il_idx])]:
            rs, nrs, rhs, sms = [], [], [], []
            for di in range(len(panels_il)):
                ev = eval_selection(smat[panels_il[di]].mean(axis=0), il_truth, 2, jits_il[di])
                rs.append(ev["regret"]); nrs.append(ev["norm_regret"]); rhs.append(ev["rho"]); sms.append(ev["severe_miss"])
            thresh_2x2_rows.append({
                "fold": "IL_14binary", "m": m_lbl, "m_num": m, "scoring": sc_name, "truth_def": "Truth_IL_BinarySheet", "subset": f"IL_{len(valid_il)}Rps",
                "regret": round(float(np.mean(rs)), 4), "mc_ci_lo": np.nan, "mc_ci_hi": np.nan,
                "norm_regret": round(float(np.mean(nrs)), 4), "rho": round(float(np.mean(rhs)), 4), "severe_miss": round(float(np.mean(sms)), 4),
            })

    # =========================================================================
    # B. ZYMOSEPTORIA TRITICI (145 isolates, 4 Regions [joint Oregon] and 5 Pops)
    # =========================================================================
    zcell = pd.read_csv(PROC / "zymo_cell_means.csv")
    ziso = pd.read_csv(PROC / "zymo_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    zhosts = sorted(zcell["host_id"].unique().tolist())
    zisos = ziso["isolate_id"].tolist()
    zpops = ziso["population"].values
 # Joint Oregon region (Oregon.R + Oregon.S collected same day/site -> 4 independent geographic regions)
    zregs = np.where(np.isin(zpops, ["Oregon.R", "Oregon.S"]), "Oregon", zpops)

    z_placl = zcell.pivot(index="isolate_id", columns="host_id", values="placl_mean").loc[zisos, zhosts].values
    z_lplacl = logit_bounded(z_placl, eps=0.025)
    z_pyc = zcell.pivot(index="isolate_id", columns="host_id", values="pycnidia_mean").loc[zisos, zhosts].values
    z_lpyc = zcell.pivot(index="isolate_id", columns="host_id", values="log1p_pycnidia_mean").loc[zisos, zhosts].values
    chk_idx = zhosts.index("4391_L")
    cand_idx = [i for i in range(len(zhosts)) if i != chk_idx]

 # Deciles with exact held-out-region min-max bands for out_of_pop_rho (SAME statistic as line) & bootstrap SD bands
    for tname, mat, is_pct in [("PLACL", z_placl, True), ("log1p_Pycnidia", z_lpyc, False)]:
        base_sd = float(np.std(mat.mean(axis=0), ddof=1))
        qb = pd.qcut(mat.mean(axis=1), 10, labels=False, duplicates="drop")
        for b in sorted(np.unique(qb)):
            ib = np.where(qb == b)[0]
            sub = mat[ib]
            b_sds = [float(np.std(mat[rng.choice(ib, size=len(ib), replace=True)].mean(axis=0), ddof=1) / base_sd) for _ in range(200)]
            o_rhos = [fast_spearman(mat[ib[zregs[ib] != rg]].mean(axis=0), mat[zregs == rg].mean(axis=0)) for rg in sorted(np.unique(zregs)) if np.any(zregs[ib] != rg)]
            dec_rows.append({
                "pathosystem": "Z. tritici", "trait": tname, "decile": int(b + 1), "n_isolates": int(len(ib)),
                "mean_virulence": round(float(sub.mean()), 3),
                "rel_host_sd": round(float(np.std(sub.mean(axis=0), ddof=1) / base_sd), 4),
                "rel_host_sd_lo": round(float(np.percentile(b_sds, 2.5)), 4), "rel_host_sd_hi": round(float(np.percentile(b_sds, 97.5)), 4),
                "ceil90_pct": round(float(100.0 * np.mean(sub >= 90.0)), 2) if is_pct else np.nan,
                "out_of_pop_rho": round(float(np.mean(o_rhos)), 4),
                "rho_lo": round(float(np.min(o_rhos)), 4), "rho_hi": round(float(np.max(o_rhos)), 4),
            })

    n_mc_z = int(CFG["n_mc_zymo"])
 # Run BOTH LORO_4regions (PRIMARY — joint Oregon holdout) and LOPO_5pops (sensitivity)
    for split_name, grp_arr in [("LORO_4regions", zregs), ("LOPO_5pops", zpops)]:
        u_grps = sorted(np.unique(grp_arr).tolist())
 # Shared CRN bank per fold
        z_crn = {g: build_shared_crn_bank(rng, np.where(grp_arr != g)[0], M_GRID, n_mc_z, len(zhosts)) for g in u_grps}

        for g in u_grps:
            te = np.where(grp_arr == g)[0]
            tr = np.where(grp_arr != g)[0]
            te_p, te_lp, te_y, te_ly = z_placl[te].mean(axis=0), z_lplacl[te].mean(axis=0), z_pyc[te].mean(axis=0), z_lpyc[te].mean(axis=0)

 # 1) Cross-endpoint comparison at m in {3, 5, 10, AllTrain} & k=2 using shared CRN panels
            for m in [3, 5, 10, 999]:
                m_lbl = "AllTrain" if m == 999 else str(m)
                panels, jits = (tr[None, :], np.zeros((1, len(zhosts)))) if m == 999 else z_crn[g][m]
                for rule in ["PLACL_only", "LogPyc_only", "RawPyc_only", "Composite_50_50"]:
                    for targ_name, targ_vec in [("PLACL", te_p), ("log1p_Pycnidia", te_ly), ("Raw_Pycnidia", te_y)]:
                        rs, nrs, rhs, sms = [], [], [], []
                        for di in range(len(panels)):
                            pk = panels[di]
                            sp_tr = z_placl[pk].mean(axis=0)
                            sly_tr = z_lpyc[pk].mean(axis=0)
                            sy_tr = z_pyc[pk].mean(axis=0)
                            if rule == "PLACL_only":
                                sc = sp_tr
                            elif rule == "LogPyc_only":
                                sc = sly_tr
                            elif rule == "RawPyc_only":
                                sc = sy_tr
                            else:
                                sc = 0.5 * (sp_tr - sp_tr.mean()) / (sp_tr.std() + 1e-9) + 0.5 * (sly_tr - sly_tr.mean()) / (sly_tr.std() + 1e-9)
                            ev = eval_selection(sc, targ_vec, 2, jits[di])
                            rs.append(ev["regret"]); nrs.append(ev["norm_regret"]); rhs.append(ev["rho"]); sms.append(ev["severe_miss"])
                        ce_rows.append({
                            "split": split_name, "fold": g, "m": m_lbl, "m_num": m, "k": 2, "rule": rule, "target": targ_name,
                            "regret": round(float(np.mean(rs)), 4), "norm_regret": round(float(np.mean(nrs)), 4),
                            "rho": round(float(np.mean(rhs)), 4), "severe_miss": round(float(np.mean(sms)), 4),
                        })

 # 2) Virulence strategies including ExtremeTopM, scale sensitivity, and model-based ranking
            for tname, scale_lbl, mat, truth_v, is_bnd in [
                ("PLACL", "raw", z_placl, te_p, True),
                ("logit_PLACL", "transformed", z_lplacl, te_lp, False),
                ("log1p_Pycnidia", "transformed", z_lpyc, te_ly, False),
                ("Raw_Pycnidia", "raw", z_pyc, te_y, False),
            ]:
                if split_name == "LOPO_5pops" and tname in ("logit_PLACL", "Raw_Pycnidia"):
                    continue  # Keep LOPO_5pops concise for main traits; full 4-trait scale grid on PRIMARY LORO_4regions
                v_tr = mat[tr].mean(axis=1)
                q25, q75 = np.percentile(v_tr, [25, 75])
                pq1, pmid, pq4 = tr[v_tr <= q25], tr[(v_tr > q25) & (v_tr < q75)], tr[v_tr >= q75]
                c_tr = mat[tr, chk_idx]
                cq25, cq75 = np.percentile(c_tr, [25, 75])
                pcq4, pcmid = tr[c_tr >= cq75], tr[(c_tr > cq25) & (c_tr < cq75)]

                q_b = {
                    "Q1_Low": build_shared_crn_bank(rng, pq1, M_VIR, n_mc_z, len(zhosts)),
                    "Q2Q3_Mid": build_shared_crn_bank(rng, pmid, M_VIR, n_mc_z, len(zhosts)),
                    "Q4_High": build_shared_crn_bank(rng, pq4, M_VIR, n_mc_z, len(zhosts)),
                    "Check4391_Q4": build_shared_crn_bank(rng, pcq4, M_VIR, n_mc_z, len(cand_idx)),
                    "Check4391_Mid": build_shared_crn_bank(rng, pcmid, M_VIR, n_mc_z, len(cand_idx)),
                }

                for m in M_VIR:
                    panels, jits = z_crn[g][m]
                    k_list = [1, 2, 3] if (split_name == "LORO_4regions" and tname in ("PLACL", "log1p_Pycnidia") and m in (3, 5)) else [2]
                    for k_val in k_list:
                        idx_ext = tr[np.argsort(v_tr)[-m:]]
                        strat_specs = [
                            ("Random", panels, jits, False, False),
                            ("Random_ModelRank", panels, jits, True, False),
                            ("Q1_Low", q_b["Q1_Low"][m][0], q_b["Q1_Low"][m][1], False, False),
                            ("Q2Q3_Mid", q_b["Q2Q3_Mid"][m][0], q_b["Q2Q3_Mid"][m][1], False, False),
                            ("Q4_High", q_b["Q4_High"][m][0], q_b["Q4_High"][m][1], False, False),
                            ("ExtremeTopM", idx_ext[None, :], np.zeros((1, len(zhosts))), False, False),
                        ]
                        if k_val == 2:
                            strat_specs += [
                                ("Check4391_Q4", q_b["Check4391_Q4"][m][0], q_b["Check4391_Q4"][m][1], False, True),
                                ("Check4391_Mid", q_b["Check4391_Mid"][m][0], q_b["Check4391_Mid"][m][1], False, True),
                            ]
                        for sname, sp_arr, sj_arr, use_mod, is_chk in strat_specs:
                            rs, nrs, rhs, sms = [], [], [], []
                            tvec = truth_v[cand_idx] if is_chk else truth_v
                            for di in range(len(sp_arr)):
                                pk = sp_arr[di]
                                if is_chk:
                                    sc = mat[pk][:, cand_idx].mean(axis=0)
                                elif use_mod:
                                    sc = model_based_host_scores(z_placl[pk], True) if tname == "PLACL" else model_based_host_scores(z_pyc[pk], False)
                                else:
                                    sc = mat[pk].mean(axis=0)
                                ev = eval_selection(sc, tvec, k_val, sj_arr[di])
                                rs.append(ev["regret"]); nrs.append(ev["norm_regret"]); rhs.append(ev["rho"]); sms.append(ev["severe_miss"])
                            fold_rows.append({"pathosystem": "Z. tritici", "trait": tname, "scale": scale_lbl, "split": split_name, "fold": g, "m": m, "k": k_val, "strategy": sname, "regret": round(float(np.mean(rs)), 4), "norm_regret": round(float(np.mean(nrs)), 4), "rho": round(float(np.mean(rhs)), 4), "severe_miss": round(float(np.mean(sms)), 4)})

 # 3) Spatial comparison (SinglePop vs Random vs MultiPop_Strat) on LORO_4regions (for log1p_Pycnidia & PLACL)
            if split_name == "LORO_4regions":
                tr_p = zpops[tr]
                p_list = sorted(np.unique(tr_p).tolist())
                for tname, mat, truth_v in [("PLACL", z_placl, te_p), ("log1p_Pycnidia", z_lpyc, te_ly)]:
                    for m in M_GRID:
                        panels, jits = z_crn[g][m]
 # Random CRN
                        r_ev = [eval_selection(mat[panels[d]].mean(axis=0), truth_v, 2, jits[d]) for d in range(n_mc_z)]
                        spatial_rows.append({"pathosystem": "Z. tritici", "trait": tname, "split": split_name, "fold": g, "m": m, "realized_m": m, "n_elig_fields": len(p_list), "strategy": "Random", "regret": round(float(np.mean([e["regret"] for e in r_ev])), 4), "norm_regret": round(float(np.mean([e["norm_regret"] for e in r_ev])), 4), "rho": round(float(np.mean([e["rho"] for e in r_ev])), 4), "severe_miss": round(float(np.mean([e["severe_miss"] for e in r_ev])), 4)})
 # SinglePop (all populations have >= 26 isolates >= 20, so exact realized_m == m)
                        sp_ev, mp_ev = [], []
                        for d in range(n_mc_z):
                            sp_p = p_list[int(rng.integers(0, len(p_list)))]
                            pool_s = tr[tr_p == sp_p]
                            sp_ev.append(eval_selection(mat[rng.choice(pool_s, size=m, replace=False)].mean(axis=0), truth_v, 2, jits[d]))
                            perm_p = rng.permutation(p_list)
                            pools = {pp: rng.permutation(tr[tr_p == pp]).tolist() for pp in perm_p}
                            pk_mp, pi = [], 0
                            while len(pk_mp) < m:
                                pp = perm_p[pi % len(perm_p)]
                                if pools[pp]:
                                    pk_mp.append(pools[pp].pop())
                                pi += 1
                            mp_ev.append(eval_selection(mat[pk_mp].mean(axis=0), truth_v, 2, jits[d]))
                        spatial_rows.append({"pathosystem": "Z. tritici", "trait": tname, "split": split_name, "fold": g, "m": m, "realized_m": m, "n_elig_fields": len(p_list), "strategy": "SinglePop_ExactM", "regret": round(float(np.mean([e["regret"] for e in sp_ev])), 4), "norm_regret": round(float(np.mean([e["norm_regret"] for e in sp_ev])), 4), "rho": round(float(np.mean([e["rho"] for e in sp_ev])), 4), "severe_miss": round(float(np.mean([e["severe_miss"] for e in sp_ev])), 4)})
                        spatial_rows.append({"pathosystem": "Z. tritici", "trait": tname, "split": split_name, "fold": g, "m": m, "realized_m": m, "n_elig_fields": len(p_list), "strategy": "MultiPop_Strat", "regret": round(float(np.mean([e["regret"] for e in mp_ev])), 4), "norm_regret": round(float(np.mean([e["norm_regret"] for e in mp_ev])), 4), "rho": round(float(np.mean([e["rho"] for e in mp_ev])), 4), "severe_miss": round(float(np.mean([e["severe_miss"] for e in mp_ev])), 4)})

 # =========================================================================
 # C. BOTRYTIS CINEREA (7 crop species, Raw + Log Scale, Strict Ancestry-Letter Holdout)
 # =========================================================================
    bls = pd.read_csv(PROC / "botrytis_lsmeans_long.csv")
    biso = pd.read_csv(PROC / "botrytis_isolates.csv")
    gmap = biso.set_index("isolate_id")["genetic_group_clean"].fillna("Unassigned").to_dict()
    crop_sp = sorted([s for s in bls["crop_species"].unique() if s != "Athaliana"])

 # Deciles ranked on the other 6 crops with bootstrap bands across crops
    b_dec_tmp = []
    for sp in crop_sp:
        piv = bls[bls["crop_species"] == sp].pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").dropna(axis=0, how="any")
        isos = piv.index.tolist()
        mat = piv.values
        base_sd = float(np.std(mat.mean(axis=0), ddof=1))
        other_v = bls[(bls["crop_species"] != sp) & (bls["crop_species"] != "Athaliana")].groupby("isolate_id")["lesion_lsmean"].mean().reindex(isos).values
        qb = pd.qcut(other_v, 10, labels=False, duplicates="drop")
        for b in sorted(np.unique(qb)):
            ib = np.where(qb == b)[0]
            ir = np.where(qb != b)[0]
            b_dec_tmp.append({
                "species": sp, "decile": int(b + 1), "n_isolates": int(len(ib)),
                "mean_virulence": float(mat[ib].mean()),
                "rel_host_sd": float(np.std(mat[ib].mean(axis=0), ddof=1) / max(base_sd, 1e-12)),
                "out_of_pop_rho": fast_spearman(mat[ib].mean(axis=0), mat[ir].mean(axis=0)),
            })
    bdf_d = pd.DataFrame(b_dec_tmp)
    for b, grp in bdf_d.groupby("decile"):
        dec_rows.append({
            "pathosystem": "B. cinerea", "trait": "lesion_other6_ranked", "decile": int(b), "n_isolates": int(round(grp["n_isolates"].mean())),
            "mean_virulence": round(float(grp["mean_virulence"].mean()), 4),
            "rel_host_sd": round(float(grp["rel_host_sd"].mean()), 4),
            "rel_host_sd_lo": round(float(grp["rel_host_sd"].min()), 4), "rel_host_sd_hi": round(float(grp["rel_host_sd"].max()), 4),
            "ceil90_pct": np.nan,
            "out_of_pop_rho": round(float(grp["out_of_pop_rho"].mean()), 4),
            "rho_lo": round(float(grp["out_of_pop_rho"].min()), 4), "rho_hi": round(float(grp["out_of_pop_rho"].max()), 4),
        })

 # Per-crop fold-level evaluation across BOTH Raw & Log scales, Strict Ancestry-Letter Holdout (A/B/C/D) & Unseen 50/50
    n_mc_b50 = int(CFG["n_mc_botrytis_5050"])
    n_mc_bg = int(CFG["n_mc_botrytis_group"])
    for sp in crop_sp:
        piv = bls[bls["crop_species"] == sp].pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").dropna(axis=0, how="any")
        isos = piv.index.tolist()
        mat_raw = piv.values
        mat_log = log_positive(mat_raw, eps=0.05)
        other_raw = bls[(bls["crop_species"] != sp) & (bls["crop_species"] != "Athaliana")].groupby("isolate_id")["lesion_lsmean"].mean().reindex(isos).values
        other_log = bls[(bls["crop_species"] != sp) & (bls["crop_species"] != "Athaliana")].assign(ly=lambda d: log_positive(d["lesion_lsmean"].values, 0.05)).groupby("isolate_id")["ly"].mean().reindex(isos).values
        groups = np.array([gmap.get(i, "Unassigned") for i in isos])

 # Split mode 1: Strict Ancestry-Letter Exclusion (hold out ALL isolates carrying letter A, B, C, or D -> ZERO shared ancestry letter)
        anc_splits = []
        for letter in ["A", "B", "C", "D"]:
            te = np.where([letter in str(g) and g != "Unassigned" for g in groups])[0]
            tr = np.where([letter not in str(g) and g != "Unassigned" for g in groups])[0]
            if len(te) >= 6 and len(tr) >= 20:
                anc_splits.append((letter, tr, te))

 # Split mode 2: Repeated 50/50 unseen isolates (CRN splits shared across scales & strategies)
        rand_splits = []
        for _ in range(n_mc_b50):
            perm = rng.permutation(len(isos))
            rand_splits.append((perm[: len(isos) // 2], perm[len(isos) // 2 :]))

        for tname, scale_lbl, mat, ov_all in [
            ("lesion_lsmean", "raw", mat_raw, other_raw),
            ("log_lesion", "transformed", mat_log, other_log),
        ]:
 # A) Strict Ancestry-Letter Exclusion (fold = crop_species, averaged over the 4 strict ancestry splits)
            for m in M_VIR:
                acc_s = {s: [] for s in ["Random", "Random_ModelRank", "Q1_Low", "Q2Q3_Mid", "Q4_High", "ExtremeTopM"]}
                for letter, tr, te in anc_splits:
                    truth = mat[te].mean(axis=0)
                    ov = ov_all[tr]
                    q25, q75 = np.percentile(ov, [25, 75])
                    p_q1, p_mid, p_q4 = tr[ov <= q25], tr[(ov > q25) & (ov < q75)], tr[ov >= q75]
                    idx_ext = tr[np.argsort(ov)[-m:]]
                    jit0 = np.zeros(mat.shape[1])
                    acc_s["ExtremeTopM"].append(eval_selection(mat[idx_ext].mean(axis=0), truth, 2, jit0))
 # Shared CRN draws within this ancestry split
                    for _ in range(n_mc_bg // 4):
                        jit = rng.uniform(0.0, 1e-9, size=mat.shape[1])
                        pr = rng.choice(tr, size=m, replace=False)
                        acc_s["Random"].append(eval_selection(mat[pr].mean(axis=0), truth, 2, jit))
                        acc_s["Random_ModelRank"].append(eval_selection(model_based_host_scores(mat_raw[pr], False), truth, 2, jit))
                        for sn, pl in [("Q1_Low", p_q1), ("Q2Q3_Mid", p_mid), ("Q4_High", p_q4)]:
                            pk = rng.choice(pl, size=min(m, len(pl)), replace=False)
                            acc_s[sn].append(eval_selection(mat[pk].mean(axis=0), truth, 2, jit))
                for sn, evs in acc_s.items():
                    fold_rows.append({"pathosystem": "B. cinerea", "trait": tname, "scale": scale_lbl, "split": "StrictAncestryOut_7crops", "fold": sp, "m": m, "k": 2, "strategy": sn, "regret": round(float(np.mean([e["regret"] for e in evs])), 4), "norm_regret": round(float(np.mean([e["norm_regret"] for e in evs])), 4), "rho": round(float(np.mean([e["rho"] for e in evs])), 4), "severe_miss": round(float(np.mean([e["severe_miss"] for e in evs])), 4)})

 # B) Unseen 50/50 isolates (per crop fold, CRN shared splits)
            for m in M_VIR:
                acc_s = {s: [] for s in ["Random", "Q1_Low", "Q2Q3_Mid", "Q4_High", "ExtremeTopM"]}
                for tr, te in rand_splits:
                    truth = mat[te].mean(axis=0)
                    ov = ov_all[tr]
                    q25, q75 = np.percentile(ov, [25, 75])
                    jit = rng.uniform(0.0, 1e-9, size=mat.shape[1])
                    acc_s["Random"].append(eval_selection(mat[rng.choice(tr, size=m, replace=False)].mean(axis=0), truth, 2, jit))
                    acc_s["Q1_Low"].append(eval_selection(mat[rng.choice(tr[ov <= q25], size=m, replace=False)].mean(axis=0), truth, 2, jit))
                    acc_s["Q2Q3_Mid"].append(eval_selection(mat[rng.choice(tr[(ov > q25) & (ov < q75)], size=m, replace=False)].mean(axis=0), truth, 2, jit))
                    acc_s["Q4_High"].append(eval_selection(mat[rng.choice(tr[ov >= q75], size=m, replace=False)].mean(axis=0), truth, 2, jit))
                    acc_s["ExtremeTopM"].append(eval_selection(mat[tr[np.argsort(ov)[-m:]]].mean(axis=0), truth, 2, jit))
                for sn, evs in acc_s.items():
                    fold_rows.append({"pathosystem": "B. cinerea", "trait": tname, "scale": scale_lbl, "split": "Unseen50_50_7crops", "fold": sp, "m": m, "k": 2, "strategy": sn, "regret": round(float(np.mean([e["regret"] for e in evs])), 4), "norm_regret": round(float(np.mean([e["norm_regret"] for e in evs])), 4), "rho": round(float(np.mean([e["rho"] for e in evs])), 4), "severe_miss": round(float(np.mean([e["severe_miss"] for e in evs])), 4)})

    df_fold_strat = pd.DataFrame(fold_rows)
    df_dec = pd.DataFrame(dec_rows)
    df_th2x2 = pd.DataFrame(thresh_2x2_rows)
    df_sh = pd.DataFrame(split_host_rows)
    df_sp = pd.DataFrame(spatial_rows)
    df_ce = pd.DataFrame(ce_rows)

    df_fold_strat.to_csv(RES / "fold_level_strategies.csv", index=False)
    df_dec.to_csv(RES / "virulence_deciles.csv", index=False)
    df_th2x2.to_csv(RES / "psojae_threshold_2x2.csv", index=False)
    df_sh.to_csv(RES / "psojae_split_host_proxy.csv", index=False)
    df_sp.to_csv(RES / "spatial_transfer_exact_m.csv", index=False)
    df_ce.to_csv(RES / "zymo_cross_endpoint.csv", index=False)
    return {
        "fold_strat": df_fold_strat, "deciles": df_dec, "thresh_2x2": df_th2x2,
        "split_host": df_sh, "spatial": df_sp, "cross_endpoint": df_ce,
    }


# ---------------------------------------------------------------------------
# 3. Genomic Covering (incl. P. sojae TRUE LOSO SSRs + B. cinerea Strict Ancestry-Out + Partial Mantel)
# ---------------------------------------------------------------------------
def run_genomics(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[3/6] Running Genomic Covering (P. sojae LOSO + 50/50, B. cinerea 50/50 + LeaveGroup + StrictAncestry) & Partial Mantel...", flush=True)
    gen_rows = []
    mantel_rows = []
    n_perm = int(CFG["n_mantel_perm"])

 # --- A) P. sojae 19 SSRs (n = 215: IN=89, OH=83, KY=43) ---
    plong = pd.read_csv(PROC / "psojae_complete_long.csv")
    ssr_df = pd.read_csv(PROC / "psojae_ssr_dist.csv", index_col=0)
    piso = pd.read_csv(PROC / "psojae_isolates.csv").set_index("isolate_id")
    rps_hosts = sorted([h for h in plong["host_id"].unique() if not h.startswith("D1_")])
    isos = list(ssr_df.index)
    pmat = plong.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[isos, rps_hosts].values
    D_ssr = ssr_df.values
    D_phen = np.sqrt(np.mean((pmat[:, None, :] - pmat[None, :, :]) ** 2, axis=2))
    locs = (piso.loc[isos, "state"].astype(str) + "::" + piso.loc[isos, "location_id"].astype(str)).values
    states = piso.loc[isos, "state"].values
    Z_same_field = (locs[:, None] == locs[None, :]).astype(float)
    Z_same_state = (states[:, None] == states[None, :]).astype(float)

    mp = mantel_and_partial(D_ssr, D_phen, [Z_same_field, Z_same_state], rng, n_perm=n_perm)
    iu = np.triu_indices(len(isos), k=1)
    ok = ~np.isnan(D_ssr[iu])
    same_l = (locs[:, None] == locs[None, :])[iu] & ok
    diff_l = (locs[:, None] != locs[None, :])[iu] & ok
    mantel_rows.append({
        "pathosystem": "P. sojae", "marker_set": "19 diploid SSR loci", **mp,
        "gen_dist_within_field": round(float(np.mean(D_ssr[iu][same_l])), 4),
        "gen_dist_between_field": round(float(np.mean(D_ssr[iu][diff_l])), 4),
        "phen_rmse_within_field": round(float(np.mean(D_phen[iu][same_l])), 2),
        "phen_rmse_between_field": round(float(np.mean(D_phen[iu][diff_l])), 2),
    })

 # Mode 1: TRUE LOSO on 215 SSR-matched isolates (IN, KY, OH) — paired against 400 CRN random draws per state
    for m in M_GEN:
        fold_r, fold_km, fold_fp = [], [], []
        for st in sorted(np.unique(states)):
            te = np.where(states == st)[0]
            tr = np.where(states != st)[0]
            truth = pmat[te].mean(axis=0)
            sub_D = D_ssr[np.ix_(tr, tr)]
            jit0 = np.zeros(len(rps_hosts))
            ev_km = eval_selection(pmat[tr[kmedoids_cover(sub_D, m)]].mean(axis=0), truth, 2, jit0)
            ev_fp = eval_selection(pmat[tr[farthest_point_medoid_init(sub_D, m)]].mean(axis=0), truth, 2, jit0)
            r_evs = [eval_selection(pmat[rng.choice(tr, size=m, replace=False)].mean(axis=0), truth, 2, rng.uniform(0.0, 1e-9, size=len(rps_hosts))) for _ in range(400)]
            r_m = float(np.mean([e["regret"] for e in r_evs]))
            fold_r.append(r_m); fold_km.append(ev_km["regret"]); fold_fp.append(ev_fp["regret"])
        for sn, arr in [("Random", fold_r), ("KMedoids_Genomic", fold_km), ("FarthestPoint_Genomic", fold_fp)]:
            diffs = np.array(arr) - np.array(fold_r)
            gen_rows.append({
                "pathosystem": "P. sojae", "split": "LOSO_3states_SSR215", "m": m, "strategy": sn,
                "regret": round(float(np.mean(arr)), 3), "fold_min": round(float(np.min(arr)), 3), "fold_max": round(float(np.max(arr)), 3),
                "paired_delta_vs_rand": round(float(np.mean(diffs)), 3), "delta_ci_lo": np.nan, "delta_ci_hi": np.nan,
                "pct_change_vs_rand": round(100.0 * float(np.mean(diffs) / max(np.mean(fold_r), 1e-9)), 1),
                "sign_flip_p": round(exact_sign_flip_p(diffs), 4) if sn != "Random" else np.nan,
            })

 # Mode 2: 200 paired 50/50 splits on 215 SSR-matched isolates
    for m in M_GEN:
        rr, rkm, rfp = [], [], []
        for _ in range(200):
            perm = rng.permutation(len(isos))
            tr, te = perm[: len(isos) // 2], perm[len(isos) // 2 :]
            truth = pmat[te].mean(axis=0)
            sub_D = D_ssr[np.ix_(tr, tr)]
            jit = rng.uniform(0.0, 1e-9, size=len(rps_hosts))
            rr.append(eval_selection(pmat[rng.choice(tr, size=m, replace=False)].mean(axis=0), truth, 2, jit)["regret"])
            rkm.append(eval_selection(pmat[tr[kmedoids_cover(sub_D, m)]].mean(axis=0), truth, 2, jit)["regret"])
            rfp.append(eval_selection(pmat[tr[farthest_point_medoid_init(sub_D, m)]].mean(axis=0), truth, 2, jit)["regret"])
        rr_a = np.array(rr)
        for sn, arr in [("Random", rr_a), ("KMedoids_Genomic", np.array(rkm)), ("FarthestPoint_Genomic", np.array(rfp))]:
            diffs = arr - rr_a
            dlo, dhi = mc_precision_ci(diffs, rng, 400)
            gen_rows.append({
                "pathosystem": "P. sojae", "split": "Unseen50_50_SSR215", "m": m, "strategy": sn,
                "regret": round(float(arr.mean()), 3), "fold_min": np.nan, "fold_max": np.nan,
                "paired_delta_vs_rand": round(float(diffs.mean()), 3), "delta_ci_lo": round(dlo, 3), "delta_ci_hi": round(dhi, 3),
                "pct_change_vs_rand": round(100.0 * float(diffs.mean() / max(rr_a.mean(), 1e-9)), 1),
                "sign_flip_p": np.nan,
            })

 # --- B) B. cinerea 271,749 SNPs (n = 93 across 7 crops) ---
    bls = pd.read_csv(PROC / "botrytis_lsmeans_long.csv")
    biso = pd.read_csv(PROC / "botrytis_isolates.csv")
    gmap = biso.set_index("isolate_id")["genetic_group_clean"].fillna("Unassigned").to_dict()
    snp_df = pd.read_csv(PROC / "botrytis_snp_dist.csv", index_col=0)
    b_isos = list(snp_df.index)
    D_snp = snp_df.values
    crop_all = bls[bls["crop_species"] != "Athaliana"].pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").loc[b_isos].values
    n_b = len(b_isos)
    D_phen_b = np.zeros((n_b, n_b), dtype=float)
    for i in range(n_b):
        for j in range(i + 1, n_b):
            ok_b = ~np.isnan(crop_all[i]) & ~np.isnan(crop_all[j])
            D_phen_b[i, j] = D_phen_b[j, i] = float(np.sqrt(np.mean((crop_all[i, ok_b] - crop_all[j, ok_b]) ** 2)))
    mb = mantel_and_partial(D_snp, D_phen_b, None, rng, n_perm=n_perm)
    mantel_rows.append({"pathosystem": "B. cinerea", "marker_set": "271,749 biallelic SNPs", **mb, "gen_dist_within_field": np.nan, "gen_dist_between_field": np.nan, "phen_rmse_within_field": np.nan, "phen_rmse_between_field": np.nan})

    crop_sp = sorted([s for s in bls["crop_species"].unique() if s != "Athaliana"])
    for split_mode in ["Unseen50_50_7crops", "LeaveGroupOut_7crops", "StrictAncestryOut_7crops"]:
        for m in M_GEN:
            crop_r, crop_km, crop_fp = [], [], []
            all_r, all_km, all_fp = [], [], []
            for sp in crop_sp:
                piv = bls[bls["crop_species"] == sp].pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").dropna(axis=0, how="any")
                common = [i for i in b_isos if i in piv.index]
                mat = piv.loc[common].values
                D_sub = snp_df.loc[common, common].values
                groups = np.array([gmap.get(i, "Unassigned") for i in common])
                if split_mode == "Unseen50_50_7crops":
                    splits = [(perm[: len(common) // 2], perm[len(common) // 2 :]) for perm in [rng.permutation(len(common)) for _ in range(100)]]
                elif split_mode == "LeaveGroupOut_7crops":
                    mg = [gg for gg, cnt in pd.Series(groups).value_counts().items() if cnt >= 6 and gg != "Unassigned"]
                    splits = [(np.where(groups != tg)[0], np.where(groups == tg)[0]) for tg in mg]
                else:
                    splits = [(np.where([ltr not in str(g) and g != "Unassigned" for g in groups])[0], np.where([ltr in str(g) and g != "Unassigned" for g in groups])[0]) for ltr in ["A", "B", "C", "D"]]

                sp_r, sp_km, sp_fp = [], [], []
                for tr, te in splits:
                    truth = mat[te].mean(axis=0)
                    sd = D_sub[np.ix_(tr, tr)]
                    jit0 = np.zeros(mat.shape[1])
                    nr_r = float(np.mean([eval_selection(mat[rng.choice(tr, size=m, replace=False)].mean(axis=0), truth, 2, rng.uniform(0.0, 1e-9, size=mat.shape[1]))["norm_regret"] for _ in range(1 if split_mode == "Unseen50_50_7crops" else 40)]))
                    nr_km = eval_selection(mat[tr[kmedoids_cover(sd, m)]].mean(axis=0), truth, 2, jit0)["norm_regret"]
                    nr_fp = eval_selection(mat[tr[farthest_point_medoid_init(sd, m)]].mean(axis=0), truth, 2, jit0)["norm_regret"]
                    sp_r.append(nr_r); sp_km.append(nr_km); sp_fp.append(nr_fp)
                    all_r.append(nr_r); all_km.append(nr_km); all_fp.append(nr_fp)
                crop_r.append(float(np.mean(sp_r))); crop_km.append(float(np.mean(sp_km))); crop_fp.append(float(np.mean(sp_fp)))

            cr_a = np.array(crop_r)
            for sn, c_arr, a_arr in [("Random", cr_a, np.array(all_r)), ("KMedoids_Genomic", np.array(crop_km), np.array(all_km)), ("FarthestPoint_Genomic", np.array(crop_fp), np.array(all_fp))]:
                cdiffs = c_arr - cr_a
                adiffs = a_arr - np.array(all_r)
                dlo, dhi = mc_precision_ci(adiffs, rng, 400)
                gen_rows.append({
                    "pathosystem": "B. cinerea", "split": split_mode, "m": m, "strategy": sn,
                    "regret": round(float(c_arr.mean()), 4), "fold_min": round(float(c_arr.min()), 4), "fold_max": round(float(c_arr.max()), 4),
                    "paired_delta_vs_rand": round(float(cdiffs.mean()), 4), "delta_ci_lo": round(dlo, 4), "delta_ci_hi": round(dhi, 4),
                    "pct_change_vs_rand": round(100.0 * float(cdiffs.mean() / max(cr_a.mean(), 1e-9)), 1),
                    "sign_flip_p": round(exact_sign_flip_p(cdiffs), 4) if sn != "Random" else np.nan,
                })

    df_gen = pd.DataFrame(gen_rows)
    df_man = pd.DataFrame(mantel_rows)
    df_gen.to_csv(RES / "genomic_sentinels.csv", index=False)
    df_man.to_csv(RES / "mantel_spatial_audit.csv", index=False)
    return df_gen, df_man


# ---------------------------------------------------------------------------
# 4. Replacement-Free Fixed-Budget Resampling and Two-Way Host x Pathogen Holdout
# ---------------------------------------------------------------------------
def run_two_way_holdout_infoopt(rng: np.random.Generator) -> pd.DataFrame:
    """Two-Way Host x Pathogen Holdout evaluating InfoOpt_IRT against Virulence Quartiles and Random.
    Sentinels are selected from (Training Pathogen Population x Reference Hosts H_ref) without
    access to unseen evaluation hosts (H_eval) or unseen target pathogen populations (I_test).
    """
    print("[4a/6] Running Two-Way Host x Pathogen Holdout (InfoOpt_IRT vs quartiles)...", flush=True)
    rows = []

 # 1) P. sojae (50 random 7-ref / 8-eval Rps splits x 3 LOSO states)
    plong = pd.read_csv(PROC / "psojae_complete_long.csv")
    piso = pd.read_csv(PROC / "psojae_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    rps_hosts = sorted([h for h in plong["host_id"].unique() if not h.startswith("D1_")])
    pisos = piso["isolate_id"].tolist()
    pstates = piso["state"].values
    p_pct = plong.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[pisos, rps_hosts].values
    u_st = sorted(np.unique(pstates).tolist())

    for m in [1, 3, 5]:
        acc = {s: [] for s in ["Random", "Q1_Low", "Q2Q3_Mid", "Q4_High", "ExtremeTopM", "InfoOpt_IRT"]}
        acc_nr = {s: [] for s in acc}
        acc_rh = {s: [] for s in acc}
        for _ in range(50):
            hp = rng.permutation(len(rps_hosts))
            h_ref, h_eval = hp[:7], hp[7:]
            for st in u_st:
                te = np.where(pstates == st)[0]
                tr = np.where(pstates != st)[0]
                truth = p_pct[te][:, h_eval].mean(axis=0)
                ref_mat = p_pct[tr][:, h_ref]
                vp = ref_mat.mean(axis=1)
                q25, q75 = np.percentile(vp, [25, 75])
                jit = rng.uniform(0.0, 1e-9, size=len(h_eval))
                for sn, pool in [("Random", tr), ("Q1_Low", tr[vp <= q25]), ("Q2Q3_Mid", tr[(vp > q25) & (vp < q75)]), ("Q4_High", tr[vp >= q75])]:
                    evs = [eval_selection(p_pct[rng.choice(pool, size=m, replace=False)][:, h_eval].mean(axis=0), truth, 2, jit) for _ in range(10)]
                    acc[sn].append(float(np.mean([e["regret"] for e in evs])))
                    acc_nr[sn].append(float(np.mean([e["norm_regret"] for e in evs])))
                    acc_rh[sn].append(float(np.mean([e["rho"] for e in evs])))
                p_ext = tr[np.argsort(vp)[-m:]]
                p_inf = tr[select_info_opt(ref_mat, m, 2, bounded_pct=True)]
                for sn, pk in [("ExtremeTopM", p_ext), ("InfoOpt_IRT", p_inf)]:
                    ev = eval_selection(p_pct[pk][:, h_eval].mean(axis=0), truth, 2, jit)
                    acc[sn].append(ev["regret"]); acc_nr[sn].append(ev["norm_regret"]); acc_rh[sn].append(ev["rho"])
        for sn in acc:
            lo, hi = mc_precision_ci(np.array(acc[sn]), rng, 300)
            rows.append({
                "pathosystem": "P. sojae", "trait": "pct_susc", "split_desc": "7_ref->8_eval_Rps x LOSO_3states", "m": m, "strategy": sn,
                "regret": round(float(np.mean(acc[sn])), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "norm_regret": round(float(np.mean(acc_nr[sn])), 4), "rho": round(float(np.mean(acc_rh[sn])), 4),
            })

 # 2) Z. tritici (50 random 6-ref / 6-eval wheat splits x 4 LORO geographic regions)
    zcell = pd.read_csv(PROC / "zymo_cell_means.csv")
    ziso = pd.read_csv(PROC / "zymo_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    zhosts = sorted(zcell["host_id"].unique().tolist())
    zisos = ziso["isolate_id"].tolist()
    zpops = ziso["population"].values
    zregs = np.where(np.isin(zpops, ["Oregon.R", "Oregon.S"]), "Oregon", zpops)
    u_rg = sorted(np.unique(zregs).tolist())
    z_placl = zcell.pivot(index="isolate_id", columns="host_id", values="placl_mean").loc[zisos, zhosts].values
    z_lpyc = zcell.pivot(index="isolate_id", columns="host_id", values="log1p_pycnidia_mean").loc[zisos, zhosts].values

    for tname, mat, is_bnd in [("PLACL", z_placl, True), ("log1p_Pycnidia", z_lpyc, False)]:
        for m in [1, 3, 5]:
            acc = {s: [] for s in ["Random", "Q1_Low", "Q2Q3_Mid", "Q4_High", "ExtremeTopM", "InfoOpt_IRT"]}
            acc_nr = {s: [] for s in acc}
            acc_rh = {s: [] for s in acc}
            for _ in range(50):
                hp = rng.permutation(len(zhosts))
                h_ref, h_eval = hp[:6], hp[6:]
                for rg in u_rg:
                    te = np.where(zregs == rg)[0]
                    tr = np.where(zregs != rg)[0]
                    truth = mat[te][:, h_eval].mean(axis=0)
                    ref_mat = mat[tr][:, h_ref]
                    vp = ref_mat.mean(axis=1)
                    q25, q75 = np.percentile(vp, [25, 75])
                    jit = rng.uniform(0.0, 1e-9, size=len(h_eval))
                    for sn, pool in [("Random", tr), ("Q1_Low", tr[vp <= q25]), ("Q2Q3_Mid", tr[(vp > q25) & (vp < q75)]), ("Q4_High", tr[vp >= q75])]:
                        evs = [eval_selection(mat[rng.choice(pool, size=m, replace=False)][:, h_eval].mean(axis=0), truth, 2, jit) for _ in range(10)]
                        acc[sn].append(float(np.mean([e["regret"] for e in evs])))
                        acc_nr[sn].append(float(np.mean([e["norm_regret"] for e in evs])))
                        acc_rh[sn].append(float(np.mean([e["rho"] for e in evs])))
                    p_ext = tr[np.argsort(vp)[-m:]]
                    p_inf = tr[select_info_opt(ref_mat, m, 2, bounded_pct=is_bnd)]
                    for sn, pk in [("ExtremeTopM", p_ext), ("InfoOpt_IRT", p_inf)]:
                        ev = eval_selection(mat[pk][:, h_eval].mean(axis=0), truth, 2, jit)
                        acc[sn].append(ev["regret"]); acc_nr[sn].append(ev["norm_regret"]); acc_rh[sn].append(ev["rho"])
            for sn in acc:
                lo, hi = mc_precision_ci(np.array(acc[sn]), rng, 300)
                rows.append({
                    "pathosystem": "Z. tritici", "trait": tname, "split_desc": "6_ref->6_eval_cultivars x LORO_4regions", "m": m, "strategy": sn,
                    "regret": round(float(np.mean(acc[sn])), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "norm_regret": round(float(np.mean(acc_nr[sn])), 4), "rho": round(float(np.mean(acc_rh[sn])), 4),
                })

 # 3) B. cinerea (Cross-Crop: 6 reference crops [72 ref hosts] -> 1 completely unseen target crop [12 eval hosts] x Strict Ancestry Holdout & 50/50)
    bls = pd.read_csv(PROC / "botrytis_lsmeans_long.csv")
    biso = pd.read_csv(PROC / "botrytis_isolates.csv")
    gmap = biso.set_index("isolate_id")["genetic_group_clean"].fillna("Unassigned").to_dict()
    crop_sp = sorted([s for s in bls["crop_species"].unique() if s != "Athaliana"])
 # Pivot each crop to isolate x host
    crop_pivs = {sp: bls[bls["crop_species"] == sp].pivot(index="isolate_id", columns="host_id", values="lesion_lsmean").dropna(axis=0, how="any") for sp in crop_sp}
    common_isos = sorted(set.intersection(*[set(p.index) for p in crop_pivs.values()]))
    groups = np.array([gmap.get(i, "Unassigned") for i in common_isos])

    for split_mode in ["CrossCrop_StrictAncestry", "CrossCrop_Unseen50_50"]:
        for m in [1, 3, 5]:
            acc_nr = {s: [] for s in ["Random", "Q1_Low", "Q2Q3_Mid", "Q4_High", "ExtremeTopM", "InfoOpt_IRT"]}
            acc_rh = {s: [] for s in acc_nr}
            for sp in crop_sp:
                mat_eval = crop_pivs[sp].loc[common_isos].values
 # Concatenate other 6 crops (72 reference host genotypes — zero overlap with sp)
                ref_mats_list = [crop_pivs[osp].loc[common_isos].values for osp in crop_sp if osp != sp]
                ref_mean_vir = np.mean([rm.mean(axis=1) for rm in ref_mats_list], axis=0)

                if split_mode == "CrossCrop_StrictAncestry":
                    splits = [(np.where([ltr not in str(g) and g != "Unassigned" for g in groups])[0], np.where([ltr in str(g) and g != "Unassigned" for g in groups])[0]) for ltr in ["A", "B", "C", "D"]]
                else:
                    splits = [(perm[: len(common_isos) // 2], perm[len(common_isos) // 2 :]) for perm in [rng.permutation(len(common_isos)) for _ in range(40)]]

                for tr, te in splits:
                    truth = mat_eval[te].mean(axis=0)
                    vp = ref_mean_vir[tr]
                    q25, q75 = np.percentile(vp, [25, 75])
 # Cross-crop InfoOpt: mean standardized Fisher signal-to-noise across the 6 reference crops ONLY
                    snr_stack = []
                    for rm in ref_mats_list:
                        sub_r = rm[tr]
                        hm = sub_r.mean(axis=0)
                        hm_c = hm - hm.mean()
                        slopes = ((sub_r - sub_r.mean(axis=1, keepdims=True)) @ hm_c) / (float(np.dot(hm_c, hm_c)) + 1e-12)
                        pred = sub_r.mean(axis=1, keepdims=True) + slopes[:, None] * hm_c[None, :]
                        mse = np.mean((sub_r - pred) ** 2, axis=1) + 1e-4
                        snr = np.where(slopes > 0, (slopes ** 2) / mse, 0.0)
                        snr_stack.append(snr / (np.mean(snr) + 1e-9))
                    info_ref = np.mean(snr_stack, axis=0)
                    p_inf = tr[np.argsort(info_ref)[-m:]]
                    p_ext = tr[np.argsort(vp)[-m:]]
                    jit = rng.uniform(0.0, 1e-9, size=mat_eval.shape[1])
                    nd = 20 if split_mode == "CrossCrop_StrictAncestry" else 1
                    for sn, pool in [("Random", tr), ("Q1_Low", tr[vp <= q25]), ("Q2Q3_Mid", tr[(vp > q25) & (vp < q75)]), ("Q4_High", tr[vp >= q75])]:
                        evs = [eval_selection(mat_eval[rng.choice(pool, size=m, replace=False)].mean(axis=0), truth, 2, jit) for _ in range(nd)]
                        acc_nr[sn].append(float(np.mean([e["norm_regret"] for e in evs])))
                        acc_rh[sn].append(float(np.mean([e["rho"] for e in evs])))
                    for sn, pk in [("ExtremeTopM", p_ext), ("InfoOpt_IRT", p_inf)]:
                        ev = eval_selection(mat_eval[pk].mean(axis=0), truth, 2, jit)
                        acc_nr[sn].append(ev["norm_regret"]); acc_rh[sn].append(ev["rho"])
            for sn in acc_nr:
                lo, hi = mc_precision_ci(np.array(acc_nr[sn]), rng, 300)
                rows.append({
                    "pathosystem": "B. cinerea", "trait": "lesion_lsmean", "split_desc": f"6_ref_crops->1_eval_crop ({split_mode})", "m": m, "strategy": sn,
                    "regret": round(float(np.mean(acc_nr[sn])), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "norm_regret": round(float(np.mean(acc_nr[sn])), 4), "rho": round(float(np.mean(acc_rh[sn])), 4),
                })

    df_2w = pd.DataFrame(rows)
    df_2w.to_csv(RES / "two_way_holdout_infoopt.csv", index=False)
    return df_2w


def run_fixed_budget(rng: np.random.Generator) -> pd.DataFrame:
    print("[4b/6] Running 100% replacement-free fixed-budget allocation (raw & block-adjusted)...", flush=True)
    rows = []

 # A) Z. tritici (4-Region Joint-Oregon LORO):
 # - All 12 cultivars: 138/145 isolates have >= 2 leaves on ALL 12 (1011_L has max 2 leaves) -> B=6 (3x2 vs 6x1) & B=12 (6x2 vs 12x1)
 # - 8 high-replication cultivars (ArinaLr34, Chinese_Spring, Drifter, Gene, Greina, Runal, Titlis, Toronit):
 # 142/145 isolates have >= 4 leaves on ALL 8 -> B=8 (2x4, 4x2, 8x1) & B=12 (3x4, 4x3, 6x2, 12x1), 100% replacement-free
    zrep = pd.read_csv(PROC / "zymo_replicates.csv")
    zcell = pd.read_csv(PROC / "zymo_cell_means.csv")
    ziso = pd.read_csv(PROC / "zymo_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    zhosts_12 = sorted(zcell["host_id"].unique().tolist())
    zhosts_8 = ["ArinaLr34", "Chinese_Spring", "Drifter", "Gene", "Greina", "Runal", "Titlis", "Toronit"]
    zisos = ziso["isolate_id"].tolist()
    zpops = ziso["population"].values
    zregs = np.where(np.isin(zpops, ["Oregon.R", "Oregon.S"]), "Oregon", zpops)

    cnt_piv = zrep.groupby(["isolate_id", "host_id"]).size().unstack(fill_value=0).loc[zisos, zhosts_12]
    elig_12 = np.where((cnt_piv[zhosts_12].values >= 2).all(axis=1))[0]
    elig_8 = np.where((cnt_piv[zhosts_8].values >= 4).all(axis=1))[0]
    print(f"  Z. tritici replacement-free pools: >=2 leaves on all 12 = {len(elig_12)}/145; >=4 leaves on 8 high-rep = {len(elig_8)}/145")

    iso_i = {x: i for i, x in enumerate(zisos)}
    host_j12 = {x: j for j, x in enumerate(zhosts_12)}
    lv_p = [[[] for _ in range(len(zhosts_12))] for _ in range(len(zisos))]
    lv_ly = [[[] for _ in range(len(zhosts_12))] for _ in range(len(zisos))]
    for r in zrep.itertuples(index=False):
        lv_p[iso_i[r.isolate_id]][host_j12[r.host_id]].append(r.placl)
        lv_ly[iso_i[r.isolate_id]][host_j12[r.host_id]].append(r.log1p_pycnidia)
    lv_p = [[np.array(x) for x in row] for row in lv_p]
    lv_ly = [[np.array(x) for x in row] for row in lv_ly]
    zmat_p = zcell.pivot(index="isolate_id", columns="host_id", values="placl_mean").loc[zisos, zhosts_12].values
    zmat_ly = zcell.pivot(index="isolate_id", columns="host_id", values="log1p_pycnidia_mean").loc[zisos, zhosts_12].values

    z_specs = [
        ("All12_cultivars", [host_j12[h] for h in zhosts_12], elig_12, [(6, [(3, 2), (6, 1)]), (12, [(6, 2), (12, 1)])]),
        ("HighRep8_cultivars", [host_j12[h] for h in zhosts_8], elig_8, [(8, [(2, 4), (4, 2), (8, 1)]), (12, [(3, 4), (4, 3), (6, 2), (12, 1)])]),
    ]
    for hset_lbl, h_idx, elig_pool, b_list in z_specs:
        for B, configs in b_list:
            for tname, lv, tmat in [("PLACL", lv_p, zmat_p), ("log1p_Pycnidia", lv_ly, zmat_ly)]:
                for m, r_l in configs:
                    fold_regs, fold_nregs, fold_rhos = [], [], []
                    for rg in sorted(np.unique(zregs)):
                        te = np.where(zregs == rg)[0]
                        tr = elig_pool[zregs[elig_pool] != rg]
                        truth = tmat[te][:, h_idx].mean(axis=0)
                        f_r, f_nr, f_rh = [], [], []
                        for _ in range(300):
                            pk = rng.choice(tr, size=m, replace=False)
                            sc = np.array([np.mean([np.mean(rng.choice(lv[ii][hj], size=r_l, replace=False)) for ii in pk]) for hj in h_idx])
                            ev = eval_selection(sc, truth, 2, rng.uniform(0.0, 1e-9, size=len(h_idx)))
                            f_r.append(ev["regret"]); f_nr.append(ev["norm_regret"]); f_rh.append(ev["rho"])
                        fold_regs.append(float(np.mean(f_r))); fold_nregs.append(float(np.mean(f_nr))); fold_rhos.append(float(np.mean(f_rh)))
                    rows.append({"pathosystem": "Z. tritici", "subset": hset_lbl, "trait": tname, "budget_B": B, "m_isolates": m, "r_leaves": r_l, "design": f"{m}x{r_l}", "regret": round(float(np.mean(fold_regs)), 4), "norm_regret": round(float(np.mean(fold_nregs)), 4), "rho": round(float(np.mean(fold_rhos)), 4), "fold_min": round(float(np.min(fold_nregs)), 4), "fold_max": round(float(np.max(fold_nregs)), 4)})

 # B) B. cinerea: 100% replacement-free across ALL 7 crops on isolates with >=3 leaves on all 12 hosts (65-97 isolates/crop)
 # -> B=6 (2x3, 3x2, 6x1) and B=12 (4x3, 6x2, 12x1) for BOTH raw leaves and Exp/Flat block-adjusted leaves
    brep = pd.read_csv(PROC / "botrytis_replicates_7sp.csv").dropna(subset=["lesion_cm2"]).copy()
    brep["flat_full"] = brep["exp"].astype(str) + "-" + brep["rep"].astype(str) + "-" + brep["flat"].astype(str)
    brep["lesion_blk_adj"] = brep["lesion_cm2"] - brep.groupby(["crop_species", "flat_full"])["lesion_cm2"].transform("mean") + brep.groupby("crop_species")["lesion_cm2"].transform("mean")
    crop_sp = sorted(brep["crop_species"].unique().tolist())

    for col_lbl, val_col in [("lesion_raw_noRepl", "lesion_cm2"), ("lesion_flatAdj_noRepl", "lesion_blk_adj")]:
        for B, configs in [(6, [(2, 3), (3, 2), (6, 1)]), (12, [(4, 3), (6, 2), (12, 1)])]:
            for m, r_l in configs:
                crop_nr, crop_rh = [], []
                for sp in crop_sp:
                    sub = brep[brep["crop_species"] == sp]
                    cnt_p = sub.groupby(["isolate_id", "host_id"]).size().unstack(fill_value=0)
                    good_isos = cnt_p[(cnt_p >= 3).all(axis=1)].index.tolist()
                    hosts = sorted(cnt_p.columns.tolist())
                    piv_m = sub.pivot_table(index="isolate_id", columns="host_id", values=val_col, aggfunc="mean").loc[good_isos, hosts].values
                    ii_m = {x: i for i, x in enumerate(good_isos)}
                    jj_m = {x: j for j, x in enumerate(hosts)}
                    lv = [[[] for _ in range(len(hosts))] for _ in range(len(good_isos))]
                    for r in sub.itertuples(index=False):
                        if r.isolate_id in ii_m and r.host_id in jj_m:
                            lv[ii_m[r.isolate_id]][jj_m[r.host_id]].append(getattr(r, val_col))
                    lv = [[np.array(x) for x in row] for row in lv]
                    s_nr, s_rh = [], []
                    for _ in range(100):
                        perm = rng.permutation(len(good_isos))
                        tr, te = perm[: len(good_isos) // 2], perm[len(good_isos) // 2 :]
                        truth = piv_m[te].mean(axis=0)
                        pk = rng.choice(tr, size=m, replace=False)
                        sc = np.array([np.mean([np.mean(rng.choice(lv[ii][j], size=r_l, replace=False)) for ii in pk]) for j in range(len(hosts))])
                        ev = eval_selection(sc, truth, 2, rng.uniform(0.0, 1e-9, size=len(hosts)))
                        s_nr.append(ev["norm_regret"]); s_rh.append(ev["rho"])
                    crop_nr.append(float(np.mean(s_nr))); crop_rh.append(float(np.mean(s_rh)))
                rows.append({"pathosystem": "B. cinerea", "subset": "All7_crops_ge3leaves", "trait": col_lbl, "budget_B": B, "m_isolates": m, "r_leaves": r_l, "design": f"{m}x{r_l}", "regret": round(float(np.mean(crop_nr)), 4), "norm_regret": round(float(np.mean(crop_nr)), 4), "rho": round(float(np.mean(crop_rh)), 4), "fold_min": round(float(np.min(crop_nr)), 4), "fold_max": round(float(np.max(crop_nr)), 4)})

    df_bud = pd.DataFrame(rows)
    df_bud.to_csv(RES / "fixed_budget_allocation.csv", index=False)
    return df_bud


# ---------------------------------------------------------------------------
# 5. Summary Fold-Level Aggregator & Sign-Flip Tests
# ---------------------------------------------------------------------------
def build_summary_comparisons(df_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold_level_strategies.csv into clean cross-fold means, ranges, and exact sign-flip P vs Random."""
    out_rows = []
    keys = ["pathosystem", "trait", "scale", "split", "m", "k"]
    for kv, grp in df_fold.groupby(keys):
        rand_by_fold = grp[grp["strategy"] == "Random"].set_index("fold")["regret"]
        rand_nr_by_fold = grp[grp["strategy"] == "Random"].set_index("fold")["norm_regret"]
        for sn, sg in grp.groupby("strategy"):
            sg_i = sg.set_index("fold")
            common = [f for f in sg_i.index if f in rand_by_fold.index]
            d_reg = sg_i.loc[common, "regret"].values - rand_by_fold.loc[common].values
            d_nr = sg_i.loc[common, "norm_regret"].values - rand_nr_by_fold.loc[common].values
            out_rows.append({
                "pathosystem": kv[0], "trait": kv[1], "scale": kv[2], "split": kv[3], "m": int(kv[4]), "k": int(kv[5]), "strategy": sn,
                "n_folds": len(sg),
                "regret_mean": round(float(sg["regret"].mean()), 4),
                "regret_fold_min": round(float(sg["regret"].min()), 4),
                "regret_fold_max": round(float(sg["regret"].max()), 4),
                "norm_regret_mean": round(float(sg["norm_regret"].mean()), 4),
                "rho_mean": round(float(sg["rho"].mean()), 4),
                "severe_miss_mean": round(float(sg["severe_miss"].mean()), 4),
                "paired_delta_vs_rand": round(float(np.mean(d_reg)), 4) if sn != "Random" else 0.0,
                "paired_norm_delta_vs_rand": round(float(np.mean(d_nr)), 4) if sn != "Random" else 0.0,
                "n_folds_better_than_rand": int(np.sum(d_reg < -1e-9)) if sn != "Random" else 0,
                "exact_sign_flip_p": round(exact_sign_flip_p(d_reg), 4) if sn != "Random" else np.nan,
            })
    df_sum = pd.DataFrame(out_rows)
    df_sum.to_csv(RES / "strategy_fold_summary.csv", index=False)
    return df_sum


def build_table4_fold_inference(res_core: dict[str, pd.DataFrame], df_gen: pd.DataFrame) -> pd.DataFrame:
    """Compute exact fold-level win counts and sign-flip P-values for every Table 4 / cross-endpoint comparison."""
    rows = []
 # 1) P. sojae 2 x 2 scoring vs Binary_70 across 3 states
    th = res_core["thresh_2x2"]
    th15 = th[th["subset"] == "All15_Rps"]
    for td in ["Truth_MeanMort", "Truth_PropVir70"]:
        for m_b, m_a, sc_b, sc_a, lbl in [
            ("3", "3", "Binary_70", "Raw_pct", "P. sojae Raw vs Bin70 (m=3)"),
            ("3", "3", "Binary_70", "Logit_Model", "P. sojae Logit vs Bin70 (m=3)"),
            ("5", "5", "Binary_70", "Raw_pct", "P. sojae Raw vs Bin70 (m=5)"),
            ("20", "5", "Binary_70", "Raw_pct", "P. sojae Raw m=5 vs Bin70 m=20"),
            ("20", "5", "Binary_70", "Logit_Model", "P. sojae Logit m=5 vs Bin70 m=20"),
        ]:
            sb = th15[(th15["truth_def"] == td) & (th15["m"] == m_b) & (th15["scoring"] == sc_b)].set_index("fold")["regret"]
            sa = th15[(th15["truth_def"] == td) & (th15["m"] == m_a) & (th15["scoring"] == sc_a)].set_index("fold")["regret"]
            folds = sorted(set(sb.index) & set(sa.index))
            d = sa.loc[folds].values - sb.loc[folds].values
            rows.append({
                "comparison": f"{lbl} [{td}]", "n_folds": len(folds),
                "base_mean": round(float(sb.loc[folds].mean()), 4), "alt_mean": round(float(sa.loc[folds].mean()), 4),
                "delta_mean": round(float(np.mean(d)), 4),
                "n_folds_alt_better": int(np.sum(d < -1e-9)), "n_folds_tie": int(np.sum(np.abs(d) <= 1e-9)),
                "exact_sign_flip_p": round(exact_sign_flip_p(d), 4),
            })

 # 2) P. sojae Equal-m SingleField vs MultiState_Strat across 3 states (m=3, 5, 10)
    sp = res_core["spatial"]
    for m_val in [3, 5, 10]:
        sb = sp[(sp["pathosystem"] == "P. sojae") & (sp["m"] == m_val) & (sp["strategy"] == "SingleField_ExactM")].set_index("fold")["regret"]
        sa = sp[(sp["pathosystem"] == "P. sojae") & (sp["m"] == m_val) & (sp["strategy"] == "MultiState_Strat")].set_index("fold")["regret"]
        folds = sorted(set(sb.index) & set(sa.index))
        d = sa.loc[folds].values - sb.loc[folds].values
        rows.append({
            "comparison": f"P. sojae MultiState vs SingleField (exact m={m_val})", "n_folds": len(folds),
            "base_mean": round(float(sb.loc[folds].mean()), 4), "alt_mean": round(float(sa.loc[folds].mean()), 4),
            "delta_mean": round(float(np.mean(d)), 4),
            "n_folds_alt_better": int(np.sum(d < -1e-9)), "n_folds_tie": int(np.sum(np.abs(d) <= 1e-9)),
            "exact_sign_flip_p": round(exact_sign_flip_p(d), 4),
        })

 # 3) Z. tritici 4-Region LORO Cross-Endpoint Composite_50_50 vs PLACL_only (m=3, 5, 10, AllTrain)
    ce = res_core["cross_endpoint"]
    ce4 = ce[(ce["split"] == "LORO_4regions") & (ce["k"] == 2)]
    for m_lbl in ["3", "5", "10", "AllTrain"]:
        for targ in ["PLACL", "log1p_Pycnidia"]:
            sb = ce4[(ce4["m"] == m_lbl) & (ce4["target"] == targ) & (ce4["rule"] == "PLACL_only")].set_index("fold")["norm_regret"]
            sa = ce4[(ce4["m"] == m_lbl) & (ce4["target"] == targ) & (ce4["rule"] == "Composite_50_50")].set_index("fold")["norm_regret"]
            folds = sorted(set(sb.index) & set(sa.index))
            d = sa.loc[folds].values - sb.loc[folds].values
            rows.append({
                "comparison": f"Z. tritici 4R Composite vs PLACL_only (m={m_lbl}, target={targ})", "n_folds": len(folds),
                "base_mean": round(float(sb.loc[folds].mean()), 4), "alt_mean": round(float(sa.loc[folds].mean()), 4),
                "delta_mean": round(float(np.mean(d)), 4),
                "n_folds_alt_better": int(np.sum(d < -1e-9)), "n_folds_tie": int(np.sum(np.abs(d) <= 1e-9)),
                "exact_sign_flip_p": round(exact_sign_flip_p(d), 4),
            })

    df_t4 = pd.DataFrame(rows)
    df_t4.to_csv(RES / "table4_fold_inference.csv", index=False)
    return df_t4


def run_infoopt_tuning_sensitivity(rng: np.random.Generator) -> pd.DataFrame:
    """Test sensitivity of InfoOpt_IRT on P. sojae 7->8 split-host LOSO across penalty and boundary_size."""
    plong = pd.read_csv(PROC / "psojae_complete_long.csv")
    piso = pd.read_csv(PROC / "psojae_isolates.csv").sort_values("isolate_id").reset_index(drop=True)
    rps_hosts = sorted([h for h in plong["host_id"].unique() if not h.startswith("D1_")])
    pisos = piso["isolate_id"].tolist()
    pstates = piso["state"].values
    p_pct = plong.pivot(index="isolate_id", columns="host_id", values="pct_susc").loc[pisos, rps_hosts].values
    u_st = sorted(np.unique(pstates).tolist())

    splits = [rng.permutation(len(rps_hosts)) for _ in range(50)]
    rows = []
    for b_size in [3, 5, 7]:
        for pen in [0.0, 0.25, 0.50]:
            regs, nregs = [], []
            for hp in splits:
                h_ref, h_eval = hp[:7], hp[7:]
                for st in u_st:
                    te = np.where(pstates == st)[0]
                    tr = np.where(pstates != st)[0]
                    truth = p_pct[te][:, h_eval].mean(axis=0)
                    L = logit_bounded(p_pct[tr][:, h_ref], eps=0.025)
                    mu = float(L.mean())
                    th = L.mean(axis=0) - mu
                    dl = L.mean(axis=1) - mu
                    p_hat = expit(mu + dl[:, None] + th[None, :])
                    b_h = np.argsort(th)[:b_size]
                    info_i = np.sum(p_hat[:, b_h] * (1.0 - p_hat[:, b_h]), axis=1)
                    resid_sd = np.std(L - (mu + dl[:, None] + th[None, :]), axis=1)
                    sc = info_i / (1.0 + pen * resid_sd)
                    pk = tr[np.argsort(sc)[-3:]]
                    ev = eval_selection(p_pct[pk][:, h_eval].mean(axis=0), truth, 2, np.zeros(len(h_eval)))
                    regs.append(ev["regret"]); nregs.append(ev["norm_regret"])
            rows.append({
                "boundary_ref_hosts": b_size, "resid_sd_penalty": pen, "m": 3,
                "regret_pct": round(float(np.mean(regs)), 4), "norm_regret": round(float(np.mean(nregs)), 4),
            })
    df_sens = pd.DataFrame(rows)
    df_sens.to_csv(RES / "infoopt_sensitivity.csv", index=False)
    return df_sens


def main() -> None:
    t0 = time.perf_counter()
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df_var, rep_audit, big_disc = run_variance_and_repeatability()
    res_core = run_unified_benchmarks(rng)
    df_gen, df_man = run_genomics(rng)
    df_2w = run_two_way_holdout_infoopt(rng)
    df_bud = run_fixed_budget(rng)
    print("[5/6] Aggregating fold-level comparisons, Table 4 inference & InfoOpt sensitivity...", flush=True)
    df_sum = build_summary_comparisons(res_core["fold_strat"])
    df_t4 = build_table4_fold_inference(res_core, df_gen)
    df_isens = run_infoopt_tuning_sensitivity(rng)

    runtime = round(time.perf_counter() - t0, 2)
    print(f"[6/6] Evaluation complete in {runtime}s", flush=True)

    # Print summary headlines
    print("\n=== Headline diagnostics ===")
    print("1) Fixed-Host Virulence and Scale Sensitivity at m=3, k=2:")
    sub_m3 = df_sum[(df_sum["m"] == 3) & (df_sum["k"] == 2) & (df_sum["split"].isin(["LOSO_3states", "LORO_4regions", "StrictAncestryOut_7crops"]))]
    print(sub_m3[["pathosystem", "trait", "scale", "strategy", "regret_mean", "norm_regret_mean", "rho_mean", "n_folds_better_than_rand", "exact_sign_flip_p"]].to_string(index=False))

    print("\n1b) Two-Way Host x Pathogen Holdout (InfoOpt_IRT vs Quartiles, m=3):")
    print(df_2w[df_2w["m"] == 3][["pathosystem", "trait", "split_desc", "strategy", "regret", "norm_regret", "rho"]].to_string(index=False))

    print("\n2) P. sojae 2 x 2 Truth-by-Scoring (All 15 Rps, mean over 3 states, m=3,5,10,20):")
    th = res_core["thresh_2x2"]
    th_p = th[(th["subset"] == "All15_Rps") & (th["m"].isin(["3", "5", "10", "20"])) & (th["scoring"].isin(["Raw_pct", "Binary_70", "Binary_Sheet", "Logit_Model"]))].groupby(["truth_def", "m_num", "scoring"])[["regret", "rho", "severe_miss"]].mean().round(3)
    print(th_p.to_string())

    print("\n3) P. sojae Split-Host Zero-Leakage Proxy (7 vs 8 Rps, 50 splits):")
    print(res_core["split_host"].to_string(index=False))

    print("\n4) P. sojae Equal-m SingleField (mean over 3 states):")
    sp_p = res_core["spatial"][res_core["spatial"]["pathosystem"] == "P. sojae"].groupby(["m", "strategy"])[["regret", "n_elig_fields"]].mean().round(2)
    print(sp_p.to_string())

    print("\n5) Z. tritici 4-Region Joint-Oregon Cross-Endpoint (m=3, 5, AllTrain):")
    ce = res_core["cross_endpoint"]
    ce_4 = ce[(ce["split"] == "LORO_4regions") & (ce["m"].isin(["3", "5", "AllTrain"])) & (ce["target"].isin(["PLACL", "log1p_Pycnidia"]))].groupby(["m", "target", "rule"])[["regret", "norm_regret", "rho"]].mean().round(3)
    print(ce_4.to_string())

    print("\n6) Genomic Covering (incl. P. sojae LOSO & B. cinerea StrictAncestryOut) & Partial Mantel:")
    print(df_man.to_string(index=False))
    print(df_gen[df_gen["m"].isin([1, 2, 3, 5])][["pathosystem", "split", "m", "strategy", "regret", "pct_change_vs_rand", "sign_flip_p"]].to_string(index=False))

    print("\n7) Replacement-Free Fixed Budget (B=8 & B=12):")
    print(df_bud.to_string(index=False))


if __name__ == "__main__":
    main()
