# Main Tables (Generated Directly From Verified Pipeline Outputs)

## Table 2. Pathosystem Scope and Sum-of-Squares Partitioning Across Three CC0-1.0 Crop Pathosystems

| Pathosystem & Trait | Scope (Isolates × Hosts) | Cell-Mean Host (%) | Cell-Mean Isolate (%) | Cell-Mean $I \times H$ (%) | $I \times H / H$ | Block SS Share (%) | Within-Cell Residual (%) | $\sigma^2_{\mathrm{err}} / \sigma^2_{I\times H}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *P. sojae* – Soybean (raw % mortality) | 455 × 15 | 45.4 | 14.5 | 40.1 | 0.88 | — | 14.3 | 1.20 |
| *P. sojae* – Soybean (70% binary call) | 455 × 15 | 39.3 | 10.9 | 49.8 | 1.26 | — | — | — |
| *Z. tritici* – Wheat (PLACL, %) | 145 × 12 | 29.5 | 26.1 | 44.4 | 1.50 | — | 46.4 | 2.54 |
| *Z. tritici* – Wheat ($\log(1 + \mathrm{pycnidia/cm}^2)$) | 145 × 12 | 53.3 | 12.6 | 34.0 | 0.64 | — | 33.4 | 1.49 |
| *B. cinerea* – 7 Eudicot Crops (mean) | 98 × 84 | 14.7 | 55.4 | 29.9 | 2.27 | 29.5 | 36.0 | 65.2 (median pos.; 4/7 crops) |

## Table 3. Out-of-Population Selection Regret and Rank Fidelity Across Sentinel Selection Strategies ($m = 3$ and $m = 5$, Top-2 Selection)

| Pathosystem & Validation Split | Strategy | $m = 3$ Native Regret (Fold Range) | $m = 3$ Norm. Regret | $m = 3$ Rank $\rho$ | Folds Better vs. Rand ($P$) | $m = 5$ Native Regret | $m = 5$ Rank $\rho$ |
|---|---|---:|---:|---:|---:|---:|---:|
| *P. sojae* (3-State LOSO, % mort.) | Intermediate (Q2–Q3) | 5.71 (5.16–6.65) | 0.070 | 0.83 | 2/3 (0.500) | 3.76 | 0.86 |
| *P. sojae* (3-State LOSO, % mort.) | Random sampling | 6.23 (4.97–7.74) | 0.076 | 0.81 | — | 4.52 | 0.86 |
| *P. sojae* (3-State LOSO, % mort.) | Random + cell logit | 5.79 (4.38–7.09) | 0.070 | 0.81 | 3/3 (0.250) | 4.02 | 0.86 |
| *P. sojae* (3-State LOSO, % mort.) | Low virulence (Q1) | 7.04 (5.37–8.40) | 0.088 | 0.83 | 1/3 (0.750) | 5.42 | 0.86 |
| *P. sojae* (3-State LOSO, % mort.) | High virulence (Q4) | 9.59 (3.19–15.56) | 0.113 | 0.69 | 1/3 (0.500) | 7.17 | 0.74 |
| *P. sojae* (3-State LOSO, % mort.) | Top-$m$ extreme | 28.15 (1.62–52.03) | 0.327 | 0.07 | 1/3 (0.500) | 27.61 | 0.22 |
| *Z. tritici* PLACL (4-Region LORO, %) | Intermediate (Q2–Q3) | 15.05 (10.12–19.59) | 0.240 | 0.68 | 2/4 (0.500) | 14.84 | 0.72 |
| *Z. tritici* PLACL (4-Region LORO, %) | Random sampling | 15.33 (10.49–19.42) | 0.245 | 0.67 | — | 14.71 | 0.72 |
| *Z. tritici* PLACL (4-Region LORO, %) | Low virulence (Q1) | 13.89 (5.69–23.73) | 0.227 | 0.65 | 2/4 (0.750) | 12.29 | 0.69 |
| *Z. tritici* PLACL (4-Region LORO, %) | High virulence (Q4) | 15.35 (9.32–19.87) | 0.242 | 0.65 | 2/4 (1.000) | 15.01 | 0.70 |
| *Z. tritici* PLACL (4-Region LORO, %) | Top-$m$ extreme | 16.07 (9.54–23.21) | 0.251 | 0.71 | 2/4 (1.000) | 16.07 | 0.73 |
| *Z. tritici* log-pyc (4-Region LORO) | Intermediate (Q2–Q3) | 0.272 (0.237–0.333) | 0.085 | 0.78 | 4/4 (0.125) | 0.252 | 0.81 |
| *Z. tritici* log-pyc (4-Region LORO) | Random sampling | 0.317 (0.257–0.362) | 0.098 | 0.76 | — | 0.300 | 0.78 |
| *Z. tritici* log-pyc (4-Region LORO) | Low virulence (Q1) | 0.240 (0.091–0.424) | 0.077 | 0.79 | 3/4 (0.375) | 0.219 | 0.82 |
| *Z. tritici* log-pyc (4-Region LORO) | High virulence (Q4) | 0.466 (0.346–0.633) | 0.139 | 0.67 | 0/4 (0.125) | 0.467 | 0.70 |
| *Z. tritici* log-pyc (4-Region LORO) | Top-$m$ extreme | 0.238 (0.154–0.335) | 0.072 | 0.67 | 3/4 (0.250) | 0.238 | 0.64 |
| *Z. tritici* log-pyc (4-Region LORO) | Check 4391_L Q4 | 0.373 (0.269–0.473) | 0.143 | 0.63 | 0/4 (0.125) | 0.325 | 0.66 |
| *B. cinerea* 7-crop (50/50, $\mathrm{cm}^2$) | Intermediate (Q2–Q3) | 0.049 (0.023–0.085) | 0.106 | 0.74 | 5/7 (0.734) | 0.033 | 0.80 |
| *B. cinerea* 7-crop (50/50, $\mathrm{cm}^2$) | Random sampling | 0.051 (0.024–0.081) | 0.108 | 0.72 | — | 0.035 | 0.79 |
| *B. cinerea* 7-crop (50/50, $\mathrm{cm}^2$) | Low virulence (Q1) | 0.056 (0.024–0.122) | 0.121 | 0.68 | 3/7 (0.500) | 0.043 | 0.74 |
| *B. cinerea* 7-crop (50/50, $\mathrm{cm}^2$) | High virulence (Q4) | 0.043 (0.018–0.088) | 0.089 | 0.72 | 6/7 (0.078) | 0.028 | 0.79 |
| *B. cinerea* 7-crop (50/50, $\mathrm{cm}^2$) | Top-$m$ extreme | 0.027 (0.011–0.057) | 0.057 | 0.75 | 7/7 (0.016) | 0.021 | 0.79 |
| *B. cinerea* 7-crop (Group excl., $\mathrm{cm}^2$) | Intermediate (Q2–Q3) | 0.043 (0.016–0.076) | 0.090 | 0.74 | 6/7 (0.047) | 0.026 | 0.80 |
| *B. cinerea* 7-crop (Group excl., $\mathrm{cm}^2$) | Random sampling | 0.051 (0.025–0.079) | 0.110 | 0.70 | — | 0.036 | 0.78 |
| *B. cinerea* 7-crop (Group excl., $\mathrm{cm}^2$) | Low virulence (Q1) | 0.060 (0.033–0.093) | 0.133 | 0.67 | 1/7 (0.141) | 0.042 | 0.74 |
| *B. cinerea* 7-crop (Group excl., $\mathrm{cm}^2$) | High virulence (Q4) | 0.059 (0.021–0.113) | 0.124 | 0.69 | 2/7 (0.109) | 0.051 | 0.74 |
| *B. cinerea* 7-crop (Group excl., $\mathrm{cm}^2$) | Top-$m$ extreme | 0.054 (0.007–0.156) | 0.109 | 0.65 | 5/7 (0.938) | 0.044 | 0.70 |

## Table 4. Operational Interventions Summary With Fold-Level Inference Where Applicable

| Category | Pathosystem & Design | Baseline | Alternative | Baseline Regret | Alternative Regret | Relative Change | Fold-Level Summary |
|---|---|---|---|---:|---:|---:|---:|
| Scoring (mean mortality target) | *P. sojae* LOSO ($m = 3$) | 70% binary call | Raw continuous $k/n$ | 14.32 pp | 6.23 pp | -56.5% | 3/3 (0.250) |
| Scoring (mean mortality target) | *P. sojae* LOSO ($m = 3$) | 70% binary call | Regularized cell logit | 14.32 pp | 5.79 pp | -59.6% | 3/3 (0.250) |
| Scoring (mean mortality target) | *P. sojae* LOSO ($m = 5$ vs. 20) | 70% binary ($m = 20$) | Raw continuous ($m = 5$) | 4.64 pp | 4.52 pp | -2.6% | 2/3 (0.750) |
| Scoring (≥70% virulent prop. target) | *P. sojae* LOSO ($m = 3$) | 70% binary call | Raw continuous $k/n$ | 11.00 pp | 4.94 pp | -55.1% | 3/3 (0.250) |
| Scoring (≥70% virulent prop. target) | *P. sojae* LOSO ($m = 3$) | 70% binary call | Regularized cell logit | 11.00 pp | 4.59 pp | -58.3% | 3/3 (0.250) |
| Scoring (≥70% virulent prop. target) | *P. sojae* LOSO ($m = 5$ vs. 20) | 70% binary ($m = 20$) | Raw continuous ($m = 5$) | 3.43 pp | 3.45 pp | +0.7% | 2/3 (1.000) |
| Dual-trait scoring | *Z. tritici* 4-Region ($m = 3$, PLACL) | PLACL only | 50/50 composite | 0.245 | 0.225 | -8.2% | 4/4 (0.125) |
| Dual-trait scoring | *Z. tritici* 4-Region ($m = 3$, log-pyc) | PLACL only | 50/50 composite | 0.116 | 0.099 | -14.8% | 4/4 (0.125) |
| Dual-trait scoring | *Z. tritici* 4-Region ($m = 5$, PLACL) | PLACL only | 50/50 composite | 0.236 | 0.217 | -8.1% | 4/4 (0.125) |
| Dual-trait scoring | *Z. tritici* 4-Region ($m = 5$, log-pyc) | PLACL only | 50/50 composite | 0.105 | 0.091 | -13.2% | 4/4 (0.125) |
| Spatial sampling | *P. sojae* LOSO (exact $m = 10$) | Single-field (7–14 fields) | Multi-state stratified | 5.78 pp | 2.39 pp | -58.6% | 3/3 (0.250) |
| Label-free genomic | *P. sojae* 19 SSRs (LOSO, $m = 3$) | Random sampling | $k$-Medoids | 5.98 pp | 4.58 pp | -23.3% | $P = 0.500$ (3 folds) |
| Label-free genomic | *P. sojae* 19 SSRs (LOSO, $m = 3$) | Random sampling | Farthest-point | 5.98 pp | 4.79 pp | -19.8% | $P = 0.750$ (3 folds) |
| Label-free genomic | *B. cinerea* 271K SNPs (50/50, $m = 3$) | Random sampling | $k$-Medoids | 0.1103 | 0.0802 | -27.3% | $P = 0.078$ (7 folds) |
| Label-free genomic | *B. cinerea* 271K SNPs (Group excl., $m = 3$) | Random sampling | Farthest-point | 0.1088 | 0.0503 | -53.7% | $P = 0.016$ (7 folds) |
| Fixed leaves ($B=12$) | *Z. tritici* 8 high-rep. cult. (PLACL) | 3 isolates × 4 leaves | 12 isolates × 1 leaf | 9.52 pp | 7.71 pp | -19.0% | 4 regions (point est.) |
| Fixed leaves ($B=12$) | *B. cinerea* 7 crops (unadj., no-repl.) | 4 isolates × 3 leaves | 12 isolates × 1 leaf | 0.1331 | 0.1296 | -2.6% | 7 crops (point est.) |

*Note:* Baseline and alternative regret values are reported in percentage points (pp) for native percentage scales (*P. sojae* mortality and *Z. tritici* fixed-leaf PLACL) and as dimensionless normalized regret (fraction of true target host range, 0 to 1) for *Z. tritici* dual-trait and *B. cinerea* rows. Relative change is expressed as a percentage (%).
