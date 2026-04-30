# RLVR Weight-Delta Analysis — Qwen3-30B-A3B

**Run compared**  
SFT start: `luqn/Qwen3-30B-A3B-Base-cp4-16h_slime-nemo-v2-rerun/iter_0002999_hf`  
RLVR end:  `shuowei/qwen3-30B-rlvr1_v2-3sv29n1u_u5j70izc_resume/iter_0000279_hf`

Both are HF-format snapshots of Qwen3-MoE (48 layers, 128 experts, hidden 2048, expert intermediate 768, GQA with 4 kv-heads × head-dim 128). 18,867 named tensors each, shapes and names identical across the two checkpoints.

The RLVR training consumed ~60 Megatron iters (219 → 279) on the `rlvr1_v2_20k_gen` recipe, 4 × 8 H100s, 7 RLVR environments (math, mcqa, code_gen, structured_outputs, calendar, reasoning_gym, instruction_following).

## TL;DR

**The RLVR update is small in magnitude but *not* particularly low-rank.** Concretely:

| Component | median ΔW ratio (‖ΔW‖_F / ‖W‖_F) | median LoRA-rank @ 90% energy | as fraction of full rank | @ 99% energy |
|---|---|---|---|---|
| attention `q_proj` | 0.25% | **1024 / 2048** | 0.50 | 0.87 |
| attention `k_proj` | 0.26% | 307 / 512 | 0.60 | 0.92 |
| attention `v_proj` | 0.22% | 342 / 512 | 0.67 | 0.95 |
| attention `o_proj` | 0.25% | **997 / 2048** | 0.49 | 0.86 |
| MoE router (gate) | 0.20% | 102 / 128 | 0.80 | 0.98 |
| MoE expert `gate_proj` | 0.25% | **476 / 768** | 0.62 | 0.92 |
| MoE expert `up_proj`   | 0.26% | 468 / 768 | 0.61 | 0.92 |
| MoE expert `down_proj` | 0.25% | 480 / 768 | 0.63 | 0.93 |
| `embed_tokens` | 0.12% | 1720 / 2048 | 0.84 | 0.98 |
| `lm_head`      | 0.12% | 1644 / 2048 | 0.80 | 0.98 |

**Cosine similarity between SFT and RLVR weight tensors is ≈ 1.0 everywhere** (never below 0.999 on any tensor).  
**Attention projections move the most (relatively), embed / lm_head the least.**  
**Layernorm drift is concentrated in early layers** (input-layernorm ~0.9% in L0, decaying to <0.01% by L47).

### Conversational summary

- RLVR moved **~0.25% of the SFT model's Frobenius energy**, spread across ~0.999-cosine directions. The checkpoints are *overwhelmingly* the SFT model.
- Unlike the classical LoRA / `LoRA-for-fine-tuning` story, **the RLVR delta is not captured well by a small number of directions**: you need roughly **half the full rank of each matrix to capture 90% of its delta**, and **~92% of full rank to capture 99%**.
- The one exception where a LoRA-style rank might make sense is the MoE router (rank-128 matrix): 80% of its ΔW energy is in the top 102 singular directions — i.e. the router is nearly-full-rank perturbed, but its full rank is so small (128) that "LoRA rank 102" would save nothing.
- Attention `q_proj` and `o_proj` (2048×4096 and 4096×2048) are the most "compressible" components: rank 1000 captures 90% of ΔW, so a LoRA rank of ~1000 over those would approximate the update faithfully. That is still a lot — a traditional LoRA might use rank 8–64.
- **No tensor in the model is strictly rank-deficient in its delta.** rank_ratio_99 ≥ 0.86 for every category.

This matches an intuition specific to RLVR on an already-well-trained model: the update is distributed broadly across the weights (many tiny corrections everywhere) rather than concentrated in a low-dimensional subspace, as is typical when you start from a specialist initialisation (LoRA-style).

## Layer-by-layer findings

Every layer is analysed independently — layer-wise data is in `layer_summary.md` (48×12 pivot tables) and `depth_trends.md` (narrative + plots). Key depth-dependent patterns:

- **Layer 0 is the strongest outlier across every metric.** Its q/o/v attention deltas are 30 – 75 % *more low-rank* than any other layer (q_proj `rank_ratio_90 = 0.30` at L0 vs ~0.50 elsewhere). A small LoRA would *only* make sense for L0.
- **`router` weight delta ramps up monotonically 5× with depth**: 0.0009 at L0 → 0.0047 at L47. Deeper routers moved much more — deeper routing decisions evidently receive far more gradient signal from the RLVR objective.
- **Expert weights show the opposite trend**: expert `up_proj` magnitude falls from 0.003 at L0 to 0.002 at L47, and expert-to-expert heterogeneity within a layer **doubles** with depth (CV of the 128-expert delta_ratio: 0.10 at L0 → 0.22 at L47).
- **Layernorms are touched only in early layers**: input-layernorm drifted 0.88 % at L0, 0.09 % at L10, essentially zero past L23. `post_attention_layernorm`, `q_norm`, `k_norm` ≈ 0 at every depth.
- **Elementwise sparsity** (fraction of ‖ΔW‖² mass held in the top-k largest-|ΔW_ij| entries): 20 – 40 % of entries carry 90 % of the energy for most tensors (a Gaussian delta would need ~46 %). Delta is moderately concentrated but not truly sparse. Router has the sparsest delta at L0 (21 % of entries cover 90 % energy) but the most uniform at L47 (41 %) — *router sparsity decreases with depth*.

See `depth_trends.md` and `plots/depth_*_trend.png` for the detailed per-layer curves.

## Files

```
qwen3-30B-rlvr1_v2_sft_vs_iter279_delta/
├── analyze_weight_delta.py      – streaming per-tensor analyzer (norms, SVD, eff. rank)
├── generate_summary.py          – overall aggregates + plots
├── depth_trends.py              – per-depth trend analysis + elementwise-sparsity pass
├── per_tensor_delta.parquet     – 18,867 rows, full spectral analysis
├── per_tensor_delta.jsonl       – same data, line-delimited
├── per_tensor_sparsity.parquet  – 18,867 rows, elementwise-concentration metrics
├── layer_summary.md             – median stats pivoted layer × category
├── category_summary.md          – overall aggregate by category
├── depth_trends.md              – layer-by-layer write-up (this is the "how does sparsity change with depth?" answer)
├── README.md                    – this file
├── full_run.log, depth_run2.log – analyzer logs
└── plots/
    ├── ratio_heatmap.png            – median delta_ratio, layer × category
    ├── rank_heatmap.png             – median rank_ratio_90, layer × category
    ├── rank_ratio_violin.png        – distribution of rank_ratio_90 per category
    ├── cumulative_energy_sample.png – cumulative Σ σ_i² / ‖ΔW‖_F² for sample tensors
    ├── depth_ratio_trend.png        – delta_ratio vs layer (one line per component)
    ├── depth_rank_trend.png         – rank_ratio_90 vs layer
    ├── depth_rank99_trend.png       – rank_ratio_99 vs layer
    └── depth_sparsity_trend.png     – elementwise concentration vs layer (50/90/99% panels)
```

The raw parquet columns are (per tensor):

| column | meaning |
|---|---|
| `name` | HF tensor name, e.g. `model.layers.12.mlp.experts.3.down_proj.weight` |
| `category` | one of `q_proj`, `k_proj`, `v_proj`, `o_proj`, `router`, `expert_gate`, `expert_up`, `expert_down`, `input_ln`, `post_attn_ln`, `q_norm`, `k_norm`, `embed`, `lm_head`, `final_norm`, `other` |
| `layer`, `expert` | integer or -1 |
| `shape`, `ndim`, `numel` | tensor shape info |
| `sft_norm`, `rlvr_norm` | `‖W‖_F` for each |
| `delta_norm` | `‖W_rlvr − W_sft‖_F` |
| `delta_ratio` | `delta_norm / sft_norm` |
| `delta_max_abs`, `sft_max_abs` | L∞ norms |
| `cosine_similarity` | flattened-vector cosine(W_sft, W_rlvr) |
| `delta_top_sv` | top-64 singular values of ΔW (list) |
| `delta_sv_count`, `delta_top1_sv` | total # of singular values, σ_max |
| `delta_stable_rank` | ‖ΔW‖_F² / σ_max² |
| `eff_rank_50` / `_90` / `_95` / `_99` | smallest rank capturing 50/90/95/99 % of `‖ΔW‖_F²` (the recommended LoRA rank at each energy threshold) |
| `rank_ratio_90` / `_99` | `eff_rank_X / min(m,n)` |
| `full_rank` | `min(m, n)` |
| `truncated_svd` | `True` if Gram-matrix path was used (max dim > 4096, i.e. `embed` and `lm_head`) |

## Methodology

### Compute

For every 2-D parameter W_sft and W_rlvr (both loaded as fp32 from the bf16 safetensors):

```
ΔW = W_rlvr − W_sft
sv = singular_values(ΔW)                              # full spectrum
eff_rank_p = argmin { k : Σ_{i≤k} σ_i² ≥ p · ‖ΔW‖_F² }
LoRA_rank_suggestion = eff_rank_0.90
```

For 1-D params (layernorms, RMS-norm scales) only the scalar norms and cosine are reported — rank is not meaningful.

### SVD strategy

- Matrices with `max(m, n) ≤ 4096` use a direct LAPACK `gesdd` SVD (economy).
- Matrices with `max(m, n) > 4096` — the **embedding (151,936 × 2,048)** and **lm_head (151,936 × 2,048)** — would take 15+ minutes single-threaded via a straight SVD. Instead we compute the Gram matrix (`ΔW.T @ ΔW`, a 2048 × 2048 matrix) via a single BLAS matmul and eigendecompose it. `sv = sqrt(eigvalsh(G))` gives the *full* singular spectrum at a fraction of the cost. The `truncated_svd` column flags rows where this path was taken. It is an exact computation (modulo fp64 eig precision), not a randomized approximation.

### Parallelism

- 48 processes via `ProcessPoolExecutor(mp_context="spawn")`.
- Each worker hard-caps BLAS thread pools to 1 (via `threadpoolctl` + env vars set before the first numpy import) so that 48 workers × 1 BLAS thread ≈ 48 cores — no contention on the 112-core pod.
- Total wall time for all 18,867 tensors: **~5 minutes**.

### What it does not do

- **No directional / basis alignment between layers or experts.** Each matrix is analysed independently. Whether the top-K left singular vectors of `ΔW_expert_0.down_proj` align with those of the SFT base weight is not examined here. If you want to test "LoRA but initialized from the SVD of ΔW as a warm-start for further RL training", the top singular vectors are in principle available — we only persisted the top-64 **singular values** (not the vectors) to keep the parquet small (8.8 MB). Recomputing the vectors is easy if needed.
- **No interpretation of *which* directions RLVR moved weights along.** A natural follow-up is to check whether the top singular vectors of ΔW align with dominant singular vectors of W_sft itself (i.e. does RLVR mostly scale existing features?) or with the gradient of the RL loss (i.e. does it trace a clear task-relevant subspace?).
- **Only one RLVR end-point is analysed** (iter_0000279). The trajectory of rank/ratio through training (e.g. every 20 iters) is more informative; this report is a single snapshot.
- **Biases are absent** in Qwen3-MoE — there are no `*_proj.bias` tensors to report on.

## How to reproduce

```bash
# 1. Make sure both HF checkpoint dirs are on NFS (sync from s3://p11-dev/... if not)
# 2. Run the analyzer on any CPU-only pod with access to the NFS:
python3 analyze_weight_delta.py --workers 48 --output-name per_tensor_delta

# 3. Aggregate and plot:
python3 generate_summary.py
```

Both scripts are self-contained — only `numpy`, `safetensors`, `pandas`, `pyarrow`, `matplotlib`, and `threadpoolctl` are required.

## Reference

The original impetus for this analysis was the observation in
[arxiv.org/abs/2602.03839v1] that successive checkpoints of a trained model exhibit
high *sparsity* in their delta. Our finding for **RLVR-on-MoE (Qwen3-30B-A3B, 60
iters of RLVR)** is different:
the delta is **small in magnitude (~0.25% of the SFT norm) but dense and not
particularly low-rank** — roughly half the full rank per matrix is needed to
capture 90% of the delta, and ~92% of full rank to capture 99%. It is *not*
cleanly compressible by a small LoRA adapter.
