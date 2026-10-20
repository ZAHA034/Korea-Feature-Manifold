
######## TOPIC: 0S natural-barrier check: do barriers make ORC on G_adj negative, and how should the 'continue' rule handle base rate and degree effects?
RECOMMENDATION:
**Main finding.** With edge length = travel time and a uniform neighbour measure, a natural barrier does not put its crossing edges among the most negative κ. Lengthening a crossing edge pushes its κ toward (1−α)(c_x/deg_x + c_y/deg_y) > 0. The negative signal lands on the flank edges instead. Under flow weighting, crossing edges do get more negative, but only because the barrier reduces flows across it. That effect cannot be separated from population gaps or 시도 effects without a flow model, and a gravity reference is fragile: a factor-2 error in the decay time gives 98–100% false rejections.

The v0.2 rule ("≥5 of top 20") therefore never fires on synthetic maps (FPR 0.00, power 0.00–0.02). With a real line set at a 15–25% base rate it would fire by chance 16–59% of the time.

**What to drop.**
- Remove "≥5 of top 20" as a decision rule.
- Do not use the hypergeometric test or OLS residual re-picking as decision rules either. Their power is ≤0.44, and residual re-picking raises the coastal share of the top 20 (0.39 → 0.48–0.50).

**Primary pre-registered test (0S, repeated in stage 3 on 표준노드링크 times).**
- **Outcome:** κ_ex = κ(d_sym) − κ_geo on G_adj, uniform measure, α = 0.5. κ_geo uses the same graph, α and measure; adjacency lengths become straight line × median(d_sym / straight line); bridge and island links keep d_sym.
- **Population:** adjacency edges only. Bridge and island links stay in the graph but are excluded from the test.
- **Index:** mean slope s = Σ|Δz| / 2 / segment length (m/km), along the straight origin-to-origin segment. Use a 90 m DEM block-averaged to 250 m, sampled every 250 m, with sea and no-data gaps linearly interpolated.
- **Covariates and regression:** none. The reference absorbs degree, boundary and length (synthetic |ρ| ≤ 0.013).
- **Test:** one-sided Spearman, H1: ρ > 0. It passes only if both the naive p and a delete-one-시도 jackknife t-test p are below 0.05 (17 시도, df 16; drop all edges touching that 시도).
- **α:** 0.05.
- **Validity check:** |ρ| < 0.1 between κ_ex and each of 1/deg_u+1/deg_v, the coast/border flag, and log straight-line length.
- **Synthetic performance:** FPR 0.00–0.02; power 0.89–0.90 at +12.5–14% crossing travel time, 1.00 from +24%.
- **Caveat:** the jackknife was simulated with 16 disjoint blocks. The 시도 version with overlapping deletion is likely more conservative, by an unmeasured amount.

**Decision rule.**
- 계속: the test passes and there is no 조정 reason.
- 조정: data/setting issues are the majority of the top 20, or results depend on island assumptions, or the validity check fails.
- 전제 재검토: no 조정 reason, but the test fails. Diagnose the cause with the first-stage ρ(min/km, s).

**Secondary reports (not used for the decision).**
1. First-stage check: ρ between minutes per km of d_sym and s.
2. The same test on commuting-OD-weighted κ, with the reference keeping the same observed weights. Note the 시도-boundary overlap when reading it.
3. The primary test repeated with excess relief h, and with mountain-line flags from two sources (산경표 대간·정맥, and the geological lines).
4. Flank-edge mean κ_ex (expected negative).
5. Structural diagnostics for κ and κ_ex, coastal share of the top 20, and island-link κ.

**Top-20 review.** Keep it descriptive only. Define a natural-barrier edge as the top 10% by excess relief h (90 m DEM, 100 m step). That fixes the base rate at 10%: expected count 2, P(X≥5) = 0.04. Report n/20 with its hypergeometric p.

**DEM.** 90 m is adequate. Excess relief is stable (rank ρ ≥ 0.976 for DEM ≤250 m and step ≤500 m). Do not use cumulative ascent itself: it is mostly length (ρ 0.86), and its level shifts 2.8× with DEM processing. Fix the processing for s before running.

**Also update.** Add a barbell bridge-length fixture to 7.1: κ = 0.2 − 0.8/L at L = 1, 2, 4, 8, 32. In DESIGN 6.4, qualify the line "κ≪0 = bottleneck" for travel-time edge lengths once ADR 0009 is adopted.

Scripts are in C:\Users\mycoms\AppData\Local\Temp\claude\c--Users-mycoms-OneDrive-Desktop-Korea-Feature-Manifold\73831373-2c49-4510-be94-e727b14555fd\scratchpad\bar (kw.py, core.py, run.py, rules.py, rules2.py, eval2.py, eval_jk.py, final_cfg.py, resstudy.py, lattice.py).

FINDINGS:

[1] (confirmed) Without a base rate, the v0.2 rule 'at least 5 of the 20 most negative κ edges are natural-barrier edges' cannot be interpreted (reviewer point 1).
  evidence: Hypergeometric test, N = 670 edges, n = 20. Chance of X ≥ 5 under random selection by base rate: 0.0002 (3%), 0.0022 (5%), 0.0406 (10%), 0.1646 (15%), 0.3697 (20%), 0.5905 (25%), 0.7663 (30%). The count needed for p < 0.05 by base rate: 3 (3%), 4 (5–6%), 5 (8–10%), 7 (15%), 8 (20%), 9 (25%), 10 (30%). Synthetic setup: 100 maps (seeds 100–199), base rate 3.1% (21 of 676 adjacency edges). Rule A had FPR 0.00 and power 0.00–0.02 in every configuration: 2 travel models × 2 neighbour measures × crossing travel time +12.5% to +141%.

[2] (refuted) Under the 0S default (uniform measure, edge length = d_sym), a natural barrier makes the edges that cross it more negative.
  evidence: Mean change Δκ on crossing edges was positive: +0.027/+0.048/+0.089/+0.125 at crossing travel time +14/+28/+70/+141% (model S), and +0.026/+0.045/+0.081/+0.114 at +12.5/+24/+55/+101% (model R). Edge length d grows faster than W1 (S, c=0.05: d +72%, W1 +57%). Limit as the crossing penalty → ∞: κ → (1−α)(c_x/deg_x + c_y/deg_y) > 0, where c_x counts x's neighbours across the barrier, including y. Checks: 2×K5 barbell bridge κ = 0.2 − 0.8/L, giving −0.6/−0.2/0/0.1/0.175 at L = 1/2/4/8/32. Triangular lattice with a straight cut: crossing edges +0.21/+0.33/+0.41 at Δ = 1/4/128. Crossing edges in the top-20 most negative fell from 0.55 to 0.03 per map.

[3] (confirmed) Under uniform measure, the barrier's negative-curvature signal sits on the 'flank' edges instead: non-crossing edges that share an endpoint with a crossing edge.
  evidence: Flank Δκ was −0.009/−0.017/−0.037/−0.063 (S, +14% to +141%), versus −0.000 to −0.006 for far edges. Flank edges in the top 20 rose from 4.06 to 5.80 per map, out of a median 61 flank edges. On the lattice, flank Δκ was −0.13 at Δ=1 and −0.51 at Δ=4, linear and unbounded. Mean κ_ex at the weakest strength: crossing +0.026, flank −0.012, far −0.002.

[4] (confirmed) With a gravity(flow)-weighted measure, crossing edges become more negative, but through reduced flow weights, not through the extra length.
  evidence: Model S, gravity weights P_y·exp(−d/30 min), crossing-edge Δκ at +14/+28/+70/+141%:
- full effect: −0.014/−0.030/−0.074/−0.119;
- weights changed alone: −0.036/−0.070/−0.154/−0.241;
- lengths changed alone: +0.022/+0.038/+0.070/+0.099.

W1 grows faster than d here (c=0.05: +85% vs +72%). Lattice net effect is non-monotone: −0.097 at Δ=4, back to −0.004 at Δ=128. Even so, only 0.82→2.34 of the top 20 were crossing edges.

[5] (refuted) Low-degree, coastal, border and island edges tend to get negative κ regardless of barriers (reviewer point 2, overall association).
  evidence: Null maps (c=0), median Spearman ρ of κ, positive in 100/100 maps unless noted:
- vs 1/deg_u+1/deg_v: +0.182 uniform (p10–p90 +0.13 to +0.22), +0.258 gravity (model R: +0.186 / +0.273);
- vs coastal flag: +0.227 / +0.292;
- vs border flag: +0.176 / +0.164;
- vs travel-time length: +0.290 / +0.287.

Island ferry links were the most positive edges under uniform: median κ +0.492, percentile 1.00. Under gravity they were +0.135 (p10–p90 −0.22 to +0.44).

[6] (confirmed) Under uniform measure, the 20 most negative κ edges are over-represented by coastal and low-degree edges (reviewer point 2, tail).
  evidence: Uniform null, top 20 versus all edges:
- coastal share 0.39 vs 0.19 (hypergeometric enrichment p<0.05 in 58% of maps);
- edges whose smaller endpoint degree is ≤ 3: 0.14 vs 0.06;
- mean 1/deg_u+1/deg_v: 0.431 vs 0.364.

Under gravity the tail is depleted instead (coastal 0.07, degree term 0.344).

[7] (refuted) Regressing κ on 1/deg_u+1/deg_v and a boundary flag (OLS), then re-picking the top 20 from residuals, removes the degree confound.
  evidence: The residual top 20 shares 17.6–17.9 of 20 edges with the raw top 20 under uniform and 14.4–16.1 under gravity. Adding log length gives 17.4–17.8 and 13.7–15.5. Coastal share of the top 20 goes up, not down: uniform 0.39 → 0.48–0.50, gravity 0.07 → 0.29–0.35. Rule E (hypergeometric on the residual top 20) reached power ≤ 0.42 only at +141% (gravity). With population avoiding mountains (N1), gravity covariate tests on the crossing flag gave FPR D 0.54/0.68 and D2 0.88/0.94 (S/R). In the pure null N0 they were 0.00/0.02.

[8] (refuted) Spearman correlation of raw κ with a DEM barrier index (max elevation or cumulative ascent) is a valid test.
  evidence: Confounded by edge length: mountain cells are big, and crossing edges are already 3.8× longer in travel time under the null (61.6 vs 16.2 min). Pure null N0 (population ignores terrain, no travel effect), uniform raw κ:
- positive-direction rejection 0.84 for excess relief, 0.89 for pass height, 1.00 for cumulative ascent;
- negative-direction rejection 0.41 for mean slope (0.56 under gravity).

After controlling degree term, boundary and log length, pass height still had 0.71 positive-direction rejections. The raw one-sided ρ<0 test with excess relief had power ≤ 0.03 in every configuration.

[9] (confirmed) Cumulative ascent is length-dependent.
  evidence: Median Spearman ρ with segment length (100 maps):
- cumulative ascent +0.864 (non-crossing +0.852);
- excess relief +0.522;
- pass height within the two cells +0.347;
- mean slope (ascent ÷ length) −0.075, or +0.002 with 250 m processing.

Cumulative ascent tracks the absolute climbing excess (ρ 0.959 S / 0.928 R) but only 0.40/0.26 for the relative excess. Mean slope reaches 0.62/0.60.

[10] (refuted) Excess relief is the better primary index.
  evidence: Excess relief beats cumulative ascent on length dependence (0.52 vs 0.86), crossing-flag AUC (0.994 vs 0.979) and DEM robustness. It still loses for the recommended test, the correlation of κ_ex with the index (joint naive + jackknife). Mean slope (250 m processing) had power 0.90/0.89 (S/R) at +14%/+12.5%, versus 0.30/0.27 for excess relief. At +28%/+24%: 1.00/1.00 versus 0.69/0.55. Keep excess relief for the descriptive classification and as a secondary index.

[11] (confirmed) The reference-differenced curvature κ_ex = κ(d_sym) − κ_geo, with uniform measure and the same graph, removes degree, boundary and length confounding and gives a calibrated, powerful test.
  evidence: κ_geo: adjacency lengths = straight line × median(d_sym / straight line); island/bridge links keep d_sym. Null correlations of κ_ex with degree term, boundary flag, log length and coast: |ρ| ≤ 0.013. Coastal share of its top 20 is 0.16–0.17 (base 0.19).

Primary test: Spearman(κ_ex, mean slope s) > 0. Pass requires naive p<0.05 and a 16-block delete-one-block jackknife t-test p<0.05.
- FPR: 0.02 (N0), 0.00 (N1-S), 0.02 (N1-R).
- Power: 0.90 (S, +14%), 0.89 (R, +12.5%), 1.00 from +24%.
- Null 5–95% range of ρ: −0.08 to +0.06. Median ρ at the weakest barrier strength: +0.118 (S) / +0.137 (R).

[12] (refuted) For flow-weighted κ, a gravity-model reference (Euclidean distance decay) is a usable control.
  evidence: Too sensitive to the decay time (true τ = 30 min). In null maps, τ_ref = 15 rejected in the positive direction 0.99–1.00 (crossing flag); τ_ref = 60 rejected in the hypothesised negative direction 0.98–1.00. With τ_ref = 30 it was calibrated (0.02). The reference that can actually be built (same observed weights, only lengths swapped) gives FPR 0.01–0.06 and power 0.88–0.90 at +12–14%. It only sees the positive length effect; the flow-suppression effect cannot be separated without a flow model.

[13] (confirmed) Under flow weighting, a population gap over mountains alone, with no travel-time effect, already makes crossing edges look negative.
  evidence: Gravity raw κ vs crossing flag, negative-direction rejection: 0.15 (N1, population ∝ exp(−z/400 m)) versus 0.00 (N0, population independent of terrain). After covariates (D2) it is 0.88 versus 0.02. A 시도 (province) administrative flow drop that partly follows mountain lines would act the same way. That case was not simulated.

[14] (confirmed) Excess relief is robust to DEM resolution and sampling interval; cumulative ascent's absolute level is not.
  evidence: 40 maps; synthetic micro-relief 18 m RMS at 0.06–6 km wavelengths; reference = 30 m DEM, 25 m step.
- Excess relief rank ρ: 0.997 (30 m/100 m), 0.994 (90 m/100 m), ≥0.976 (DEM ≤250 m, step ≤500 m), 0.93 (1 km), 0.88 (2 km). Median relative error 1.5% (90 m/100 m), 9% (1 km).
- Cumulative ascent, crossing-edge median: 2,437 m (30 m/25 m), 1,500 m (90 m/100 m), 878 m (1 km/1 km). Rank stays ≥0.95 only because it is dominated by length.
- Mean slope, power at the weakest strength (naive, S/R): 0.82/0.75 at 30 m/25 m; 0.90/0.97 at 90 m/100 m; 1.00/0.97 at 250 m/250 m. Micro-relief adds uniform per-km ascent and dilutes the contrast: crossing/other mean slope 52.1/47.5 m/km at 30 m versus 23.3/15.5 at 250 m.

[15] (confirmed) The choice of mountain-line source changes the result of the binary 'crosses mountain line' flag.
  evidence: Adding three low ridges (250–400 m, mimicking 정맥 lines) to the line set raised the base rate from 3.1% to 5.5% (median). The κ_ex test with the flag then lost power: 0.59 → 0.43 (S, +14%), 0.87 → 0.66 (R, +12.5%), 1.00 → 0.93 (S, +28%).

[16] (uncertain) These synthetic power numbers and DEM settings carry over to real Korea, which has tunnel-rich mountain roads, OSM free-flow speeds and 시도-level flow effects.
  evidence: Not verified. The barrier was modelled as a climbing penalty (S: 1.3·L·exp(0.12N)/60 km/h + c·mean ascent on a ≥2 km DEM; R: 1 km raster with anisotropic least cost). Real crossing-edge excess travel time is unknown. Province flow suppression, urban speed differences and real DEM roughness were not simulated. The sign result for the uniform measure is analytic (the limit formula) and does not depend on the synthetic world; the magnitudes do. Scripts and results: C:\Users\mycoms\AppData\Local\Temp\claude\c--Users-mycoms-OneDrive-Desktop-Korea-Feature-Manifold\73831373-2c49-4510-be94-e727b14555fd\scratchpad\bar

######## TOPIC: Making the hyperbolic-vs-Euclidean judgment independent of the iteration budget: hyperbolic radius cap R_max plus a two-stage convergence rule (ADR 0010 decisions 4 and 7, PLAN 6.4 row 3-3 and closing paragraph, PLAN 7.1 tree protocol)
RECOMMENDATION:
The reviewer's diagnosis holds: without a cap the tree distortion keeps falling with iterations (median mrd 0.0537 at 2k, 0.0149 at 100k), so a ratio at a fixed iteration count measures the budget, not the data. A cap alone does not fix it. Adam with the plain 1e-6/200 rule still stops on oscillation (up to 9.5% above the optimum on a near-flat grid), which would trip the existing 1e-3 failure guard. An lr-halving variant also stopped 0.11% above E in geoopt. Recommended pre-registration:

1. **Cap.** Hyperbolic radius cap R_max = 5 (curvature -1): after every RiemannianAdam step, project points with d_P(0,x) > 5 back toward the origin (|x| <= tanh(2.5) = 0.9866). No cap for the Euclidean space.
2. **Two-stage optimization, same for both spaces.**
   - Adam (lr 0.05). Stop when the best-so-far relative stress improves by <= 1e-6 (relative) over 200 iterations, or at 50,000.
   - Then a scipy L-BFGS-B polish (ftol 1e-12, max 20,000). For the hyperbolic space use x = tanh(rho/2)*w/|w| with box rho in [0, R_max].
   - Judge only on polished values, using the minimum-stress restart whose polish converged. A "no hyperbolic advantage (flat limit)" record (within 1e-3 of E) needs no convergence.
3. **No fixed curvature or scale.** It produces a false disadvantage of 1.2-9.2x on planar grids and worse local minima on the tree.
4. **Fix 0.8, R_max = 5 and the optimizer rule together, before 0S-4.** R_max moves the tree ratio 0.441 / 0.225 / 0.115 at 3 / 5 / 8.
5. **New tree fixture rule.** Seeds 0-4; pass iff best ratio <= 0.3, median ratio <= 0.8, E best = 0.1506 +- 0.003, and H best = 0.0339 +- 0.001 with converged polish. This passes for 99.9855% of all 2,118,760 five-seed subsets (reference ratio 0.2253, about 1 min in geoopt).

Consistency edits outside the requested blocks:
- ADR 0010 status line: the cap and optimizer rule become pre-0S-4 items.
- PLAN 1장 "바로 할 일".
- PLAN 4.2, ADR 0010 row.
- PLAN 6.1a 0S-4 row and the 규모 line (2,000 iterations is no longer the budget).
- Last sentence of the 6.4 "서술 규칙" bullet.
- Optionally, add "반지름 상한 / 척도 고정 / 반복 수 고정" to ADR 0010 선택지.

Not verified: behaviour on real n≈250 inputs and the geoopt runtime at n=250 (see the 'uncertain' finding). No project files were modified. Scripts and raw results are in C:\Users\mycoms\AppData\Local\Temp\claude\c--Users-mycoms-OneDrive-Desktop-Korea-Feature-Manifold\73831373-2c49-4510-be94-e727b14555fd\scratchpad\rcap: capemb.py, driver.py, analyzeA.py, polish_checks.py, the A_*/h8_* .jsonl files and selfcheck.json. The environment is that folder's sibling fixtures71\.venv-torch.

FINDINGS:

[1] (confirmed) Without a radius cap, relative stress with a learned scale has no finite optimum on tree data, so a fixed-iteration H/E ratio is an artifact of the budget.
  evidence: Setup: tree depth 4 (31 nodes, hop distances), 2D Poincare ball (curvature -1); numpy mirror of geoopt RiemannianAdam (lr 0.05, float64); closed-form s* each step; init U(+-0.1)^2; seeds 0-9; only geoopt's float64 clamp (radius 12.21). Median mrd: 0.0667 / 0.0537 / 0.0390 / 0.0268 / 0.0185 / 0.0160 at 1k / 2k / 5k / 10k / 20k / 50k iterations, and 0.0149 at 100k. Median max radius grows 3.40 -> 8.81. 0/10 runs met the 1e-6/200 stop rule within 50k. A bounded L-BFGS-B continuation reached 0.0106 and was still moving. The H/E ratio (E = 0.1506) drifts 0.357 (2k) -> 0.106 (50k) -> 0.099 (100k).

[2] (confirmed) A hard cap d_P(0,x) <= R_max, applied as radial projection after every step (|x| <= tanh(R_max/2)), gives a finite optimum and a reproducible converged value.
  evidence: Theory: the capped feasible set is compact, so the minimum is either attained or equals the flat-limit (Euclidean) value. R_max=5, 50 seeds, Adam + bounded L-BFGS-B (ftol 1e-12): 40/50 seeds land on mrd 0.033938482-0.033938561 (stress 1.042766781-1.042766782). All 50 polishes report CONVERGENCE (247-520 its). Polishing from 2x the Adam iterations changes mrd by <= 1.7e-7. Optimum geometry: root at the origin, 4/31 points (leaves) at the cap, hyperbolic diameter 10.000, s* = 0.8015. Adam-only best-of-10 is 0.0339 for every lr (0.02/0.05/0.1), tol (1e-5/1e-6/1e-7) and window (200/500). numpy vs geoopt RiemannianAdam+cap: |dmrd| <= 6.9e-7 after 3,000 steps. The old fixed-T protocol was reproduced exactly (median 0.0537, range 0.0528-0.2036; E 0.1506).

[3] (refuted) The plain stop rule alone (Adam, relative improvement < 1e-6 over 200 iterations, 50k cap) yields a budget-independent value.
  evidence: (1) grid_road (15x15 jittered towns, 8-neighbour roads, travel time, n=225), R5, seeds 0-9: Adam oscillates (stress 30.19<->32.20). The rule fires at 1,238-2,093 its with stress up to 9.5% above the constrained local minimum 29.662734 (E = 29.758673). That would trip ADR 0010's H > E(1+1e-3) failure guard. For seed 0, a 2x budget moved mrd 0.024845 -> 0.025393. (2) lr halving on plateau (x8) fixed numpy (29.6633 at k=14,424), but geoopt's trajectory stopped at k=4,323 at 29.79059 = 1.00107 x E. (3) Depth-7 tree (255 nodes), R5, seed 2: the rule never fired by 50k (stress 329.49; 328.86 at 100k). A final bounded L-BFGS-B polish fixes all three: all 10 grid_road seeds reach 29.662734 (spread 1.5e-12, <= 1,277 its, <= 5.6 s); the geoopt end point reaches 29.662734 (760 its, 0.95 s); tree7 seed 2 reaches 328.59 (4,769 its, 7.3 s).

[4] (confirmed) R_max changes the verdict, so it must be pre-registered; 5 is the best compromise (float64-safe, below float32's 6.21, finite optimum, tree advantage kept).
  evidence: Tree 2D, best mrd / H/E ratio / points at cap: R2 0.0961 / 0.638 / 8 of 31; R3 0.0664 / 0.441 / 8; R4 0.0465 / 0.309 / 8; R5 0.0339 / 0.225 / 4; R6 0.0260 / 0.173 / 4; R8 0.0173 / 0.115 / 4. Precision: the geoopt float32 projection radius is 2*artanh(1-4e-3) = 6.2126 (float64: 12.2061). A float32 run with R_max=8 stopped at radius 6.2126 (mrd 0.0248, not the float64 0.0173). R_max=5 in float32 gives 0.033943-0.033956, the same as float64. At R5, 1-|x| >= 0.01339. R8 search is slow: 3/10 tree seeds and 5/5 depth-7 seeds reached 50k without the rule firing, and depth-7 values moved 1.4e-3 to 9.8e-2 between 50k and 100k.

[5] (confirmed) Under every cap, planar metrics show no false hyperbolic advantage (ratio about 1).
  evidence: n=225, seeds 0-9. R in {2,3,5,8} gives identical results because the cap never binds (0/225 points at cap). grid_road: H 0.024822 vs E 0.025008, ratio 0.9926 (stress ratio 0.9968), max radius 0.39-0.46. grid_l1 (Manhattan): the polish drives the layout to the flat limit (max radius 0.001); stress 357.649125 vs E 357.649117 (ratio 1.0000), inside the ADR's 1e-3 flat-limit guard. 2/10 polishes hit 20,000 its at the same value. The flat limit lies inside the capped set: shrinking the E optimum to radius 0.1 / 0.01 gives stress gaps of 1.07e-3 / 1.06e-5. On an exactly Euclidean 15x15 grid, E reaches about 1e-16 and H creeps toward the flat limit (mrd 4.2-4.5e-4), so the ratio is meaningless there (degenerate case).

[6] (confirmed) (c) Road-like metric with fast hub corridors shows only a mild advantage, and the cap is irrelevant.
  evidence: grid_road_hw (capital -> 4 regional -> 16 sub-hub expressways, 100 km/h): H 0.058621 vs E 0.062145 for all R (ratio 0.9433, stress ratio 0.9064). Max radius 1.227, 0 points at cap. Polish 66-79 its; seed spread 4.7e-15.

[7] (refuted) A fixed scale (s = diam/D_H, i.e. fixed curvature) is an adequate alternative to the cap.
  evidence: The optimum is finite, but the flat limit is excluded. Planar H/E ratios for D_H 4 / 6 / 10 / 16: grid_road 2.29 / 3.95 / 6.83 / 9.21; grid_road_hw 1.28 / 1.89 / 2.94 / 3.61; grid_l1 1.18 / 1.40 / 1.94 / 2.59 (5 seeds each). This is a structural false disadvantage, and it also defeats the 1e-3 flat-limit guard. Tree: D_H=10 (about R5's optimum, s 0.80 vs 0.8015) reaches 0.0339 in only 3/10 seeds (median 0.0984 vs 0.0339 under the cap). D_H=16 best is 0.1082 vs 0.0173 under cap 8.

[8] (confirmed) Tree-fixture thresholds under the capped, converged protocol pass robustly across seed subsets.
  evidence: Protocol: R5, Adam + polish, seeds 0-4. Rule: {best-stress ratio <= 0.3, median ratio <= 0.8, E best = 0.1506 +- 0.003, H best = 0.0339 +- 0.001 with converged polish}. Over all C(50,5) = 2,118,760 subsets: 99.9855% pass. The 308 failures are 252 all-local-H subsets plus 56 all-local-E subsets. Worst median ratio 0.6576; 'all 5 paired H<E' holds in 90.0%. Seeds 0-4: H 0.099059 / 0.033939 x4, E 0.150633 / 0.188722 / 0.188722 / 0.150633 / 0.150633; best and median ratio both 0.2253. Basins over 50 seeds: H 0.0339 x40, 0.0991 x8, 0.1951 x2 (seeds 7 and 45, whose Adam hit 50k); E 0.1506 x42, 0.1887 x5, 0.227 x1, 0.2955 x2. geoopt + polish, seeds 0-4: H Adam 9,405-11,898 its + polish 235-474 its; E 361-508 its; 54 s total under background load. The old thresholds (0.5/0.8/guard) also pass 99.9855% at R5, but they would also accept R3 (0.441) and R4 (0.309).

[9] (confirmed) At the real-data size (255-node binary tree), R5 keeps a clear advantage and the two-stage protocol converges.
  evidence: E 0.23783 (Adam 647-769 its). R3: Adam 3,298-5,951 + polish 133-262 its; best 0.12497 (ratio 0.525). R5: Adam rule fired 4/5 (13,283-31,017 its; seed 2 hit 50,000); polish converged 5/5 (1,311-4,769 its, <= 11 s); best 0.06486 (ratio 0.2727), median ratio 0.3374; polishing from 2x Adam changes mrd by <= 6.6e-9; 84/255 points at cap for the best restart.

[10] (confirmed) The cap and polish also work in 5 and 10 dimensions (PLAN 3-3 uses dims 2, 5, 10).
  evidence: Tree only, seeds 0-4, R5: 5D best 0.02528 vs E 0.09304 (ratio 0.272), 16/31 points at cap; 10D best 0.02529 vs E 0.09277 (ratio 0.273). Uncapped runs never converge (about 0.0152 at 50k). The general (rho, w) polish in 5D took 3,990 its (0.47 s, 0.025289 -> 0.025281), and in 2D it reproduces the polar-coordinate polish to 1e-12.

[11] (uncertain) Real KFM inputs (n about 250, d_time / d_flow) will behave like the synthetic cases (Adam <= 50k, polish converges, cap rarely binds for d_time).
  evidence: Not run on real data. Synthetic n=225-255: Adam stopped at 1.2k-31k its (one depth-7 seed at the 50k cap); polish took 66-4,769 its, and the flat-limit L1 grid hit the 20,000 cap. geoopt cost at n=250 under this protocol was not measured (PLAN 6.1a: 2,000 its in 17 s).
