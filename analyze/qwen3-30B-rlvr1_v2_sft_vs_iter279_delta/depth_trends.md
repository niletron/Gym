# Depth trends — how the RLVR delta varies across transformer layers
This document answers two questions, using the 18,867-tensor analysis:
1. **Do the layers differ?** Yes, and the pattern is consistent and interpretable — detailed below.
2. **How does sparsity change with depth?** Both definitions of sparsity — spectral (low-rank-ness) and elementwise (most deltas are tiny) — show systematic trends with depth, presented below.

## 1. Magnitude of change (`delta_ratio = ‖ΔW‖_F / ‖W_sft‖_F`) across depth
Median across all tensors of that category at each layer (128 experts collapsed to median for `expert_*`).

- **q_proj**: early (L0–3)=0.002605, mid (L20–27)=0.002521, late (L44–47)=0.002461
- **k_proj**: early (L0–3)=0.002657, mid (L20–27)=0.002548, late (L44–47)=0.003308
- **v_proj**: early (L0–3)=0.002757, mid (L20–27)=0.002241, late (L44–47)=0.002113
- **o_proj**: early (L0–3)=0.002652, mid (L20–27)=0.002544, late (L44–47)=0.002449
- **router**: early (L0–3)=0.00121, mid (L20–27)=0.002007, late (L44–47)=0.004416
- **expert_gate**: early (L0–3)=0.002668, mid (L20–27)=0.00262, late (L44–47)=0.002231
- **expert_up**: early (L0–3)=0.002798, mid (L20–27)=0.002629, late (L44–47)=0.002082
- **expert_down**: early (L0–3)=0.002557, mid (L20–27)=0.002613, late (L44–47)=0.002141

**Notable depth patterns in `delta_ratio`:**
- `router` weight ramps up 5× going deeper: 0.0009 at L0 → 0.0047 at L47. Deeper routers got adjusted much more than shallow ones.
- `k_proj` and `v_proj` are largely flat across depth except for a small late-layer uptick in `k_proj` (0.0029 at L47, ~0.0025 in middle).
- `q_proj` and `o_proj` are almost depth-invariant (std ≈ 0.0001 on ratios of ~0.0025).
- Expert matrices (`gate_proj`, `up_proj`, `down_proj`) slightly *decrease* with depth (from ~0.003 in L0 to ~0.002 in L47) — opposite to the router.
- **Layer 0 is always the biggest outlier**: 5–50 % larger ratio than the mean for `q_proj`, `v_proj`, `o_proj`, `expert_*`, plus a large layernorm jump.

## 2. Spectral sparsity (`rank_ratio_90 = eff_rank_90 / min(m,n)`) across depth
`rank_ratio_90` is the smallest LoRA-rank (as a fraction of full rank) that captures 90 % of the delta's Frobenius energy. *Lower = more low-rank = "more LoRA-compressible"*.

- **q_proj**: early (L0–3)=0.4521, mid (L20–27)=0.4994, late (L44–47)=0.4944
- **k_proj**: early (L0–3)=0.584, mid (L20–27)=0.5986, late (L44–47)=0.5591
- **v_proj**: early (L0–3)=0.6147, mid (L20–27)=0.6616, late (L44–47)=0.6538
- **o_proj**: early (L0–3)=0.4788, mid (L20–27)=0.4704, late (L44–47)=0.4896
- **router**: early (L0–3)=0.8105, mid (L20–27)=0.792, late (L44–47)=0.7793
- **expert_gate**: early (L0–3)=0.591, mid (L20–27)=0.6131, late (L44–47)=0.6348
- **expert_up**: early (L0–3)=0.5807, mid (L20–27)=0.6016, late (L44–47)=0.6336
- **expert_down**: early (L0–3)=0.6191, mid (L20–27)=0.6208, late (L44–47)=0.6248

**Notable depth patterns in `rank_ratio_90`:**
- **L0 is dramatically more low-rank than deeper layers**: q_proj `rank_ratio_90 = 0.30` at L0 vs ~0.50 elsewhere (roughly twice as compressible). Same pattern in `o_proj` (0.38 vs ~0.49) and `v_proj` (0.50 vs ~0.67). The first transformer block's RLVR update is genuinely the most LoRA-friendly.
- Deeper in the stack, rank_ratio stays roughly flat in a band (attn Q/O around 0.48-0.53, K around 0.60, V around 0.66-0.69, expert matrices 0.61-0.64).
- Very-last layers (L44-L47) show a mild **drop in rank_ratio for K and O** (K 0.52→0.56, O 0.42-0.48) — slightly more low-rank near the output than in the middle — but the effect is much smaller than the L0 outlier.
- `router` is the least low-rank component overall (rank_ratio_90 ≈ 0.80 at every depth): the 128-dim router is perturbed almost fully.

## 3. Elementwise sparsity (concentration of ΔW on few entries) across depth
For each tensor, sort all |ΔW_ij| in descending order and record the fraction of entries needed to capture 50 / 90 / 99 % of the Frobenius energy ‖ΔW‖_F². **Low value = sparse** (energy is concentrated in a few large entries); **value → 1 = fully uniform** (every entry contributes equally, like a Gaussian).

### 50 % of energy: fraction of entries needed (lower = sparser)
- **q_proj**: early (L0–3)=0.08224, mid (L20–27)=0.09456, late (L44–47)=0.09468
- **k_proj**: early (L0–3)=0.08848, mid (L20–27)=0.0961, late (L44–47)=0.1093
- **v_proj**: early (L0–3)=0.08878, mid (L20–27)=0.08447, late (L44–47)=0.08295
- **o_proj**: early (L0–3)=0.09676, mid (L20–27)=0.09422, late (L44–47)=0.09148
- **router**: early (L0–3)=0.06673, mid (L20–27)=0.1068, late (L44–47)=0.1116
- **expert_gate**: early (L0–3)=0.09033, mid (L20–27)=0.09339, late (L44–47)=0.0822
- **expert_up**: early (L0–3)=0.09383, mid (L20–27)=0.09386, late (L44–47)=0.07721
- **expert_down**: early (L0–3)=0.0901, mid (L20–27)=0.09349, late (L44–47)=0.07992

### 90 % of energy
- **q_proj**: early (L0–3)=0.2748, mid (L20–27)=0.2858, late (L44–47)=0.2963
- **k_proj**: early (L0–3)=0.289, mid (L20–27)=0.2929, late (L44–47)=0.3458
- **v_proj**: early (L0–3)=0.2938, mid (L20–27)=0.2602, late (L44–47)=0.258
- **o_proj**: early (L0–3)=0.3003, mid (L20–27)=0.2876, late (L44–47)=0.2804
- **router**: early (L0–3)=0.2085, mid (L20–27)=0.359, late (L44–47)=0.4099
- **expert_gate**: early (L0–3)=0.2821, mid (L20–27)=0.2827, late (L44–47)=0.2544
- **expert_up**: early (L0–3)=0.2899, mid (L20–27)=0.2821, late (L44–47)=0.2412
- **expert_down**: early (L0–3)=0.2747, mid (L20–27)=0.2815, late (L44–47)=0.2482

### 99 % of energy
- **q_proj**: early (L0–3)=0.4195, mid (L20–27)=0.4209, late (L44–47)=0.437
- **k_proj**: early (L0–3)=0.43, mid (L20–27)=0.4323, late (L44–47)=0.5063
- **v_proj**: early (L0–3)=0.4378, mid (L20–27)=0.3893, late (L44–47)=0.3843
- **o_proj**: early (L0–3)=0.4389, mid (L20–27)=0.4241, late (L44–47)=0.4131
- **router**: early (L0–3)=0.3137, mid (L20–27)=0.5283, late (L44–47)=0.6268
- **expert_gate**: early (L0–3)=0.411, mid (L20–27)=0.4115, late (L44–47)=0.373
- **expert_up**: early (L0–3)=0.4231, mid (L20–27)=0.412, late (L44–47)=0.3538
- **expert_down**: early (L0–3)=0.4006, mid (L20–27)=0.4112, late (L44–47)=0.3643

**Notable depth patterns in elementwise concentration:**
- `q_proj`: L0 = 0.269, L47 = 0.302 — less concentrated (denser) at the deep end
- `k_proj`: L0 = 0.298, L47 = 0.317 — ≈ flat
- `v_proj`: L0 = 0.367, L47 = 0.271 — more concentrated (sparser) at the deep end
- `o_proj`: L0 = 0.344, L47 = 0.308 — more concentrated (sparser) at the deep end
- `router`: L0 = 0.157, L47 = 0.407 — less concentrated (denser) at the deep end
- `expert_gate`: L0 = 0.294, L47 = 0.251 — more concentrated (sparser) at the deep end
- `expert_up`: L0 = 0.296, L47 = 0.238 — more concentrated (sparser) at the deep end
- `expert_down`: L0 = 0.275, L47 = 0.254 — ≈ flat

For reference, a Gaussian (perfectly isotropic delta) would need ~46 % of entries to capture 90 % of energy. All our categories require 20 – 40 %, so the ΔW is somewhat more concentrated than Gaussian but still not a true sparse signal (true sparse means < 5 % of entries).

## 4. Layernorm drift vs. depth
Layernorms are 1-D, so no rank analysis. Magnitudes of `input_layernorm.weight` delta:

| layer bucket | median input_ln delta_ratio |
|---|---|
| L0-L7 | 0.0018 |
| L8-L15 | 0.0009 |
| L16-L23 | 0.0005 |
| L24-L31 | 0.0001 |
| L32-L39 | 0.0001 |
| L40-L47 | 0.0000 |

Layernorm change is **sharply front-loaded**: L0 input-layernorm drifted **0.88 %**, decaying to essentially zero by the output side. `post_attention_layernorm` and q_norm/k_norm are all ≈ 0 throughout (model barely touched them).

## 5. Expert-to-expert heterogeneity within a layer
Does RLVR hit all 128 experts equally in a given layer, or do some experts absorb more change?

| layer | 128-expert delta_ratio mean | std | CV = std/mean | min | max |
|---|---|---|---|---|---|
| L0 | 0.0029 | 0.0003 | 0.100 | 0.0022 | 0.0034 |
| L5 | 0.0025 | 0.0003 | 0.105 | 0.0017 | 0.0038 |
| L15 | 0.0026 | 0.0003 | 0.131 | 0.0014 | 0.0044 |
| L23 | 0.0026 | 0.0004 | 0.139 | 0.0012 | 0.0032 |
| L35 | 0.0025 | 0.0004 | 0.141 | 0.0011 | 0.0032 |
| L47 | 0.0021 | 0.0005 | 0.224 | 0.0003 | 0.0029 |

**Expert spread grows with depth**: CV (coeff. of variation) doubles from 0.10 at L0 to 0.22 at L47. Later layers have *substantially more heterogeneity* across experts — a few experts absorb much more RLVR change than the median.

## Summary: depth-dependent patterns

- **Layer 0 is a strong outlier in every dimension** — biggest attention delta, biggest layernorm delta, and most low-rank delta (rank_ratio_90 at L0 is 60 – 75 % of what it is at any other layer). This is consistent with RLVR predominantly re-adjusting the input-side feature extractor.
- **Router weight delta grows monotonically 5× with depth** — deeper routers are moved far more than shallow ones, suggesting that routing decisions in later layers are more task-specific and therefore receive more gradient signal.
- **Expert weights have the opposite trend**: they slightly shrink with depth, and become more heterogeneous across the 128 experts within a layer.
- **Neither spectral sparsity (rank_ratio_90) nor elementwise sparsity is uniform with depth.** L0 is the lone layer where a small LoRA would make any sense; everywhere else the delta is too high-rank.
- **Layernorms are touched only in early layers** (L0 input-ln: 0.88 %, decaying to ~0 by L15).
