"""
Step 3: Two-stage 200-breeding-line phase-diagram and shared-latent observation-model simulation.
Evaluates sentinel-isolate selection beyond 12-15 differential/cultivar host panels without host overlap:
  - Stage 1 (Historical Reference Panel): N_TR = 80 candidate isolates screened on H_HIST = 20
    reference check lines. All sentinel rules (Random, Intermediate Q2-Q3, Aggressive Q4,
    and Fisher-Information Optimal InfoOpt_IRT) select m = 3 sentinels exclusively from those
    20 reference lines without access to the 200 unseen breeding lines.
  - Stage 2 (Prospective Screening): Those m = 3 sentinels inoculate H_NEW = 200 unseen,
    closely spaced breeding lines (fresh host effects, fresh I*H, fresh assay error),
    selecting the top k = 10 (top 5%) lines.
  - Stage 3 (Target Population Evaluation): Selected lines are scored by normalized regret
    against an unseen target pathogen population of N_TE = 100 isolates, reporting within-model
    paired Monte Carlo standard errors across N_REPS = 120 replications (Morris et al. 2019).

Three simulation components:
  1. Bounded Binomial-Logit Grid: mean severity mu_p in {0.15..0.85} x sigma2_err/sigma2_IH in {0.25..16}
     with 10 binomial trials per cell (illustrates floor vs. ceiling transition).
  2. Observation Models With Shared Latent Draws: shares latent host (sigma_h = 0.70),
     interaction (sigma_IH = 0.45), isolate (sigma_I = 1.00), and standard-normal noise draws
     across continuous bounded logit-normal models (mu_p = 0.25, 0.85; error added on logit scale
     before inverse-logit, without an additional binomial draw) and unbounded log-scale models
     (latent linear predictor scaled by 0.5 before exponentiation, mean-dependent noise proportional
     to mu^alpha for alpha in {0.50, empirical median from botrytis_noise_scaling.csv, 1.00},
     truncated at zero).
  3. Unbounded Sub-Linear Noise Grid: isolate dispersion sigma_I in {0.25..1.05} x error ratio,
     parameterized directly by the empirical median cell-level mean-SD exponent read from
     results/botrytis_noise_scaling.csv.

Outputs:
  - results/simulation_phase_diagram_200hosts.csv
  - results/simulation_controlled_link_contrast.csv
All paths are relative to the project root.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"
CFG = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))

SEED = int(CFG["seed"]) + 777
H_HIST = 20                      # historical reference panel used strictly to select sentinels
H_NEW = int(CFG["n_sim_hosts"])  # 200 completely unseen closely spaced breeding lines
K_SIM = 10                       # top 5% (10 of 200 breeding lines)
M_SIM = 3                        # 3-isolate sentinel panel
N_TR = 80                        # candidate isolate bank size
N_TE = 100                       # unseen target pathogen population
N_REPS = int(CFG["n_sim_reps"])  # Monte Carlo replications per grid cell


def eval_top_k_regret(scores: np.ndarray, truth: np.ndarray, k: int) -> float:
    sel = np.argsort(scores)[:k]
    orc = np.argsort(truth)[:k]
    t_range = float(np.max(truth) - np.min(truth))
    return float((np.mean(truth[sel]) - np.mean(truth[orc])) / max(t_range, 1e-12))


def load_empirical_botrytis_alpha() -> tuple[float, float, float]:
    """Read fitted cell-level mean-SD power-law scaling exponent directly from botrytis_noise_scaling.csv."""
    p = RES / "botrytis_noise_scaling.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run code/02_sentinel_evaluation.py first.")
    df = pd.read_csv(p)
    med = float(df["cell_alpha_slope"].median())
    lo = float(df["cell_alpha_slope"].min())
    hi = float(df["cell_alpha_slope"].max())
    return med, lo, hi


def _summarize_cell(reg_rand, reg_mid, reg_q4, reg_info, meta: dict) -> dict:
    a_r, a_m, a_q, a_i = map(np.asarray, (reg_rand, reg_mid, reg_q4, reg_info))
    means = {
        "Random": float(a_r.mean()),
        "Intermediate_Q2Q3": float(a_m.mean()),
        "Aggressive_Q4": float(a_q.mean()),
        "InfoOpt_IRT": float(a_i.mean()),
    }
    dq = a_q - a_m
    di = a_i - a_r
    se_dq = float(np.std(dq, ddof=1) / np.sqrt(len(dq)))
    se_di = float(np.std(di, ddof=1) / np.sqrt(len(di)))
    winner = min(means, key=means.get)
    sorted_vals = sorted(means.values())
    return {
        **meta,
        **{f"nreg_{kk}": round(vv, 4) for kk, vv in means.items()},
        "q4_minus_mid": round(float(dq.mean()), 4),
        "mc_se_q4_minus_mid": round(se_dq, 4),
        "info_minus_rand": round(float(di.mean()), 4),
        "mc_se_info_minus_rand": round(se_di, 4),
        "margin_1st_vs_2nd": round(float(sorted_vals[1] - sorted_vals[0]), 4),
        "best_rule": winner,
    }


def run_bounded_grid(rng: np.random.Generator) -> pd.DataFrame:
    """Two-stage bounded binomial-logit phase diagram (10 trials/cell): mean severity mu_p x sigma2_err / sigma2_IH."""
    mu_p_grid = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
    ratio_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_h = 0.90
    sigma_i = 1.25
    sigma_ih = 0.65
    n_plants = 10

    rows = []
    for mu_p in mu_p_grid:
        mu_l = float(np.log(mu_p / (1.0 - mu_p)))
        for ratio in ratio_grid:
            sigma_err = sigma_ih * float(np.sqrt(ratio))
            reg_rand, reg_mid, reg_q4, reg_info = [], [], [], []
            for _ in range(N_REPS):
                th_hist = rng.normal(0.0, sigma_h, size=H_HIST)
                th_new = rng.normal(0.0, sigma_h, size=H_NEW)
                delta_tr = rng.normal(0.0, sigma_i, size=N_TR)
                delta_te = rng.normal(0.0, sigma_i, size=N_TE)

                eta_hist = mu_l + th_hist[None, :] + delta_tr[:, None] + rng.normal(0.0, sigma_ih, size=(N_TR, H_HIST)) + rng.normal(0.0, sigma_err, size=(N_TR, H_HIST))
                y_hist = rng.binomial(n_plants, expit(eta_hist)) / float(n_plants)

                v_hist = y_hist.mean(axis=1)
                q25, q75 = np.percentile(v_hist, [25, 75])
                p_mid = np.where((v_hist > q25) & (v_hist < q75))[0]
                p_q4 = np.where(v_hist >= q75)[0]

                p_sm = (y_hist * n_plants + 0.25) / (n_plants + 0.5)
                L_h = np.log(p_sm / (1.0 - p_sm))
                l_mu = float(L_h.mean())
                th_hat = L_h.mean(axis=0) - l_mu
                dl_hat = L_h.mean(axis=1) - l_mu
                p_hat = expit(l_mu + th_hat[None, :] + dl_hat[:, None])
                res_tail = np.argsort(th_hat)[:5]
                info_i = np.sum(p_hat[:, res_tail] * (1.0 - p_hat[:, res_tail]), axis=1)
                idx_info = np.argsort(info_i)[-M_SIM:]

                idx_r = rng.choice(N_TR, size=M_SIM, replace=False)
                idx_m = rng.choice(p_mid, size=M_SIM, replace=False)
                idx_q = rng.choice(p_q4, size=M_SIM, replace=False)

                eta_new = mu_l + th_new[None, :] + delta_tr[:, None] + rng.normal(0.0, sigma_ih, size=(N_TR, H_NEW)) + rng.normal(0.0, sigma_err, size=(N_TR, H_NEW))
                y_new = rng.binomial(n_plants, expit(eta_new)) / float(n_plants)

                p_te = expit(mu_l + th_new[None, :] + delta_te[:, None] + rng.normal(0.0, sigma_ih, size=(N_TE, H_NEW)))
                truth = p_te.mean(axis=0)

                reg_rand.append(eval_top_k_regret(y_new[idx_r].mean(axis=0), truth, K_SIM))
                reg_mid.append(eval_top_k_regret(y_new[idx_m].mean(axis=0), truth, K_SIM))
                reg_q4.append(eval_top_k_regret(y_new[idx_q].mean(axis=0), truth, K_SIM))
                reg_info.append(eval_top_k_regret(y_new[idx_info].mean(axis=0), truth, K_SIM))

            rows.append(_summarize_cell(reg_rand, reg_mid, reg_q4, reg_info, {
                "scale_regime": "Bounded_BinomialLogit", "x_param_name": "mean_disease_prob", "x_val": mu_p,
                "err_to_ih_ratio": ratio, "alpha_noise": np.nan, "H_hist": H_HIST, "H_new": H_NEW, "k": K_SIM, "m": M_SIM,
            }))
    return pd.DataFrame(rows)


def run_controlled_link_contrast(rng: np.random.Generator, alpha_emp: float) -> pd.DataFrame:
    """Observation-model comparison with shared latent draws (sigma_h = 0.70, sigma_IH = 0.45, sigma_I = 1.00)
    across continuous bounded logit-normal models and unbounded log-scale models with alpha in {0.50, alpha_emp, 1.00}."""
    ratio_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_h = 0.70
    sigma_ih = 0.45
    sigma_i = 1.00

    link_specs = [
        ("Bounded_Logit_mu025", "bounded", 0.25, np.nan),
        ("Bounded_Logit_mu085", "bounded", 0.85, np.nan),
        ("Unbounded_Log_alpha050", "unbounded", np.nan, 0.50),
        ("Unbounded_Log_alphaEmp", "unbounded", np.nan, alpha_emp),
        ("Unbounded_Log_alpha100", "unbounded", np.nan, 1.00),
    ]
    rows = []
    for ratio in ratio_grid:
        # Pre-generate shared latent host/isolate/I*H and base normal noise draws across observation models
        lat_draws = []
        for _ in range(N_REPS):
            lat_draws.append((
                rng.normal(0.0, sigma_h, size=H_HIST),
                rng.normal(0.0, sigma_h, size=H_NEW),
                rng.normal(0.0, sigma_i, size=N_TR),
                rng.normal(0.0, sigma_i, size=N_TE),
                rng.normal(0.0, sigma_ih, size=(N_TR, H_HIST)),
                rng.normal(0.0, sigma_ih, size=(N_TR, H_NEW)),
                rng.normal(0.0, sigma_ih, size=(N_TE, H_NEW)),
                rng.normal(0.0, 1.0, size=(N_TR, H_HIST)),
                rng.normal(0.0, 1.0, size=(N_TR, H_NEW)),
            ))

        for link_name, kind, mu_p, alpha in link_specs:
            reg_rand, reg_mid, reg_q4, reg_info = [], [], [], []
            for th_h, th_n, d_tr, d_te, ih_h, ih_n, ih_te, z_eh, z_en in lat_draws:
                if kind == "bounded":
                    mu_l = float(np.log(mu_p / (1.0 - mu_p)))
                    s_err = sigma_ih * float(np.sqrt(ratio))
                    # Continuous logit-normal observation model (error added on logit scale before inverse-logit)
                    y_hist = expit(mu_l + th_h[None, :] + d_tr[:, None] + ih_h + s_err * z_eh)
                    y_new = expit(mu_l + th_n[None, :] + d_tr[:, None] + ih_n + s_err * z_en)
                    truth = expit(mu_l + th_n[None, :] + d_te[:, None] + ih_te).mean(axis=0)
                else:
                    # Exact same latent eta_ih = th + d + ih; scale Gaussian error to match exact variance ratio
                    base_ih_sd = float(np.sqrt(np.exp(sigma_ih ** 2) - 1.0))
                    base_err_sd = base_ih_sd * float(np.sqrt(ratio))
                    mu_h = np.exp(0.5 * (th_h[None, :] + d_tr[:, None] + ih_h))
                    mu_n = np.exp(0.5 * (th_n[None, :] + d_tr[:, None] + ih_n))
                    y_hist = np.clip(mu_h + base_err_sd * (mu_h ** alpha) * z_eh, 0.0, None)
                    y_new = np.clip(mu_n + base_err_sd * (mu_n ** alpha) * z_en, 0.0, None)
                    truth = np.exp(0.5 * (th_n[None, :] + d_te[:, None] + ih_te)).mean(axis=0)

                v_hist = y_hist.mean(axis=1)
                q25, q75 = np.percentile(v_hist, [25, 75])
                p_mid = np.where((v_hist > q25) & (v_hist < q75))[0]
                p_q4 = np.where(v_hist >= q75)[0]

                if kind == "bounded":
                    p_sm = np.clip(y_hist, 0.01, 0.99)
                    L_h = np.log(p_sm / (1.0 - p_sm))
                    lm = float(L_h.mean())
                    th_hat, dl_hat = L_h.mean(axis=0) - lm, L_h.mean(axis=1) - lm
                    p_hat = expit(lm + th_hat[None, :] + dl_hat[:, None])
                    res_tail = np.argsort(th_hat)[:5]
                    idx_info = np.argsort(np.sum(p_hat[:, res_tail] * (1.0 - p_hat[:, res_tail]), axis=1))[-M_SIM:]
                else:
                    hm = y_hist.mean(axis=0)
                    hm_c = hm - hm.mean()
                    slopes = ((y_hist - y_hist.mean(axis=1, keepdims=True)) @ hm_c) / (float(np.dot(hm_c, hm_c)) + 1e-12)
                    pred = y_hist.mean(axis=1, keepdims=True) + slopes[:, None] * hm_c[None, :]
                    mse_i = np.mean((y_hist - pred) ** 2, axis=1) + 1e-4
                    idx_info = np.argsort(np.where(slopes > 0, (slopes ** 2) / mse_i, -1e9))[-M_SIM:]

                idx_r = rng.choice(N_TR, size=M_SIM, replace=False)
                idx_m = rng.choice(p_mid, size=M_SIM, replace=False)
                idx_q = rng.choice(p_q4, size=M_SIM, replace=False)

                reg_rand.append(eval_top_k_regret(y_new[idx_r].mean(axis=0), truth, K_SIM))
                reg_mid.append(eval_top_k_regret(y_new[idx_m].mean(axis=0), truth, K_SIM))
                reg_q4.append(eval_top_k_regret(y_new[idx_q].mean(axis=0), truth, K_SIM))
                reg_info.append(eval_top_k_regret(y_new[idx_info].mean(axis=0), truth, K_SIM))

            rows.append(_summarize_cell(reg_rand, reg_mid, reg_q4, reg_info, {
                "link_regime": link_name, "err_to_ih_ratio": ratio,
                "sigma_h": sigma_h, "sigma_ih": sigma_ih, "sigma_i": sigma_i,
                "alpha_noise": round(alpha, 4) if not np.isnan(alpha) else np.nan,
            }))
    return pd.DataFrame(rows)


def run_unbounded_grid(rng: np.random.Generator, alpha_emp: float) -> pd.DataFrame:
    """Two-stage unbounded multiplicative lesion phase diagram parameterized directly by empirical
    median B. cinerea cell-level mean-SD exponent alpha_emp read from results/botrytis_noise_scaling.csv."""
    sigma_i_grid = [0.25, 0.45, 0.65, 0.85, 1.05]
    ratio_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    sigma_h = 0.35
    sigma_ih = 0.25

    rows = []
    for sig_i in sigma_i_grid:
        for ratio in ratio_grid:
            base_ih_sd = float(np.sqrt(np.exp(sigma_ih ** 2) - 1.0))
            base_err_sd = base_ih_sd * float(np.sqrt(ratio))
            reg_rand, reg_mid, reg_q4, reg_info = [], [], [], []
            for _ in range(N_REPS):
                th_hist = rng.normal(0.0, sigma_h, size=H_HIST)
                th_new = rng.normal(0.0, sigma_h, size=H_NEW)
                delta_tr = rng.normal(0.0, sig_i, size=N_TR)
                delta_te = rng.normal(0.0, sig_i, size=N_TE)

                mu_hist = np.exp(th_hist[None, :] + delta_tr[:, None] + rng.normal(0.0, sigma_ih, size=(N_TR, H_HIST)))
                y_hist = np.clip(mu_hist + rng.normal(0.0, base_err_sd * (mu_hist ** alpha_emp)), 0.0, None)

                v_hist = y_hist.mean(axis=1)
                q25, q75 = np.percentile(v_hist, [25, 75])
                p_mid = np.where((v_hist > q25) & (v_hist < q75))[0]
                p_q4 = np.where(v_hist >= q75)[0]

                hm = y_hist.mean(axis=0)
                hm_c = hm - hm.mean()
                slopes = ((y_hist - y_hist.mean(axis=1, keepdims=True)) @ hm_c) / (float(np.dot(hm_c, hm_c)) + 1e-12)
                pred = y_hist.mean(axis=1, keepdims=True) + slopes[:, None] * hm_c[None, :]
                mse_i = np.mean((y_hist - pred) ** 2, axis=1) + 1e-4
                idx_info = np.argsort(np.where(slopes > 0, (slopes ** 2) / mse_i, -1e9))[-M_SIM:]

                idx_r = rng.choice(N_TR, size=M_SIM, replace=False)
                idx_m = rng.choice(p_mid, size=M_SIM, replace=False)
                idx_q = rng.choice(p_q4, size=M_SIM, replace=False)

                mu_new = np.exp(th_new[None, :] + delta_tr[:, None] + rng.normal(0.0, sigma_ih, size=(N_TR, H_NEW)))
                y_new = np.clip(mu_new + rng.normal(0.0, base_err_sd * (mu_new ** alpha_emp)), 0.0, None)

                y_te = np.exp(th_new[None, :] + delta_te[:, None] + rng.normal(0.0, sigma_ih, size=(N_TE, H_NEW)))
                truth = y_te.mean(axis=0)

                reg_rand.append(eval_top_k_regret(y_new[idx_r].mean(axis=0), truth, K_SIM))
                reg_mid.append(eval_top_k_regret(y_new[idx_m].mean(axis=0), truth, K_SIM))
                reg_q4.append(eval_top_k_regret(y_new[idx_q].mean(axis=0), truth, K_SIM))
                reg_info.append(eval_top_k_regret(y_new[idx_info].mean(axis=0), truth, K_SIM))

            rows.append(_summarize_cell(reg_rand, reg_mid, reg_q4, reg_info, {
                "scale_regime": "Unbounded_SublinearNoise", "x_param_name": "isolate_virulence_sd", "x_val": sig_i,
                "err_to_ih_ratio": ratio, "alpha_noise": round(alpha_emp, 4), "H_hist": H_HIST, "H_new": H_NEW, "k": K_SIM, "m": M_SIM,
            }))
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.perf_counter()
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    alpha_emp, alpha_lo, alpha_hi = load_empirical_botrytis_alpha()
    print(f"Loaded empirical B. cinerea cell mean-SD exponent from botrytis_noise_scaling.csv: median={alpha_emp:.4f} (range {alpha_lo:.3f}-{alpha_hi:.3f})", flush=True)
    print(f"Running two-stage 200-host simulations (H_hist={H_HIST} -> H_new={H_NEW}, k={K_SIM}, m={M_SIM}, reps={N_REPS})...", flush=True)
    df_b = run_bounded_grid(rng)
    df_c = run_controlled_link_contrast(rng, alpha_emp)
    df_u = run_unbounded_grid(rng, alpha_emp)
    df_all = pd.concat([df_b, df_u], ignore_index=True)
    df_all.to_csv(RES / "simulation_phase_diagram_200hosts.csv", index=False)
    df_c.to_csv(RES / "simulation_controlled_link_contrast.csv", index=False)
    print(f"Simulations finished in {time.perf_counter() - t0:.1f}s")
    print("\n=== OBSERVATION MODELS WITH SHARED LATENT DRAWS (Q4 - Q2Q3, +/- within-model MC SE) ===")
    print(df_c.pivot(index="err_to_ih_ratio", columns="link_regime", values="q4_minus_mid").round(4).to_string())
    print("\n=== BOUNDED GRID (Q4 - Q2Q3) ===")
    print(df_b.pivot(index="err_to_ih_ratio", columns="x_val", values="q4_minus_mid").round(3).to_string())
    print("\n=== UNBOUNDED EMPIRICAL-ALPHA GRID (Q4 - Q2Q3) ===")
    print(df_u.pivot(index="err_to_ih_ratio", columns="x_val", values="q4_minus_mid").round(3).to_string())


if __name__ == "__main__":
    main()
