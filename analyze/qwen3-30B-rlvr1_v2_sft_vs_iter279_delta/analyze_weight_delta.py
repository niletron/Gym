import os as _os

# Force single-threaded BLAS in every process that imports this module. Must happen
# BEFORE numpy is imported so the BLAS thread-pool is initialised at that size.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

"""
analyze_weight_delta.py

Compares two HF-format Qwen3-MoE checkpoints (SFT "start" vs RLVR "end") and
computes per-tensor delta statistics:

  - delta_norm       = ||W_rlvr - W_sft||_F
  - delta_ratio      = ||ΔW||_F / ||W_sft||_F                 (relative change)
  - cosine_similarity between flattened W_sft and W_rlvr
  - For 2D tensors: SVD(ΔW) → singular values → effective rank at
    50 / 90 / 95 / 99 % cumulative Frobenius energy (squared singular values).
    The "LoRA rank" suggestion is eff_rank_90 (adjustable).
  - Stable rank = ||ΔW||_F² / σ_max²  (a smooth rank proxy)

Output: parquet + jsonl with one row per tensor in the model.

Assumes tensor name parity between SFT and RLVR (HF → HF conversion).
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import multiprocessing as mp
import numpy as np
from safetensors import safe_open


SFT_DIR = "/shared/dev/luqn/checkpoints/Qwen3-30B-A3B-Base-cp4-16h_slime-nemo-v2-rerun/iter_0002999_hf"
RLVR_DIR = "/shared/dev/shuowei/niletron/exp/qwen3-30B-rlvr1_v2-3sv29n1u_u5j70izc_resume/iter_0000279_hf"
OUT_DIR = "/shared/dev/shuowei/niletron/rlvr/analyze/qwen3-30B-rlvr1_v2_sft_vs_iter279_delta"
TOP_SV = 64  # number of singular values to store per tensor for plotting


# ------------ categorization helpers ------------

def categorize(name):
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "experts" in name and "down_proj" in name:
        return "expert_down"
    if "experts" in name and "gate_proj" in name:
        return "expert_gate"
    if "experts" in name and "up_proj" in name:
        return "expert_up"
    if name.endswith("mlp.gate.weight") or re.search(r"mlp\.gate\.weight$", name):
        return "router"
    if "q_proj" in name:
        return "q_proj"
    if "k_proj" in name:
        return "k_proj"
    if "v_proj" in name:
        return "v_proj"
    if "o_proj" in name:
        return "o_proj"
    if "input_layernorm" in name:
        return "input_ln"
    if "post_attention_layernorm" in name:
        return "post_attn_ln"
    if "q_norm" in name:
        return "q_norm"
    if "k_norm" in name:
        return "k_norm"
    if name == "model.norm.weight":
        return "final_norm"
    return "other"


def layer_idx(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def expert_idx(name):
    m = re.search(r"experts\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def effective_ranks(sv):
    """Given singular values (descending), return ranks at 50/90/95/99% cumulative
    Frobenius energy (energy = sv**2)."""
    e = (sv.astype(np.float64)) ** 2
    total = float(e.sum())
    if total <= 0:
        return 0, 0, 0, 0
    cum = np.cumsum(e) / total
    def r_at(th):
        # smallest k (1-indexed) with cum[k-1] >= th
        return int(np.searchsorted(cum, th) + 1)
    return r_at(0.5), r_at(0.9), r_at(0.95), r_at(0.99)


def stable_rank(sv):
    if sv[0] <= 0:
        return 0.0
    return float(((sv.astype(np.float64)) ** 2).sum() / (float(sv[0]) ** 2))


# ------------ core analysis ------------

def analyze_tensor(name, W_sft, W_rlvr):
    delta = W_rlvr - W_sft
    sft_norm = float(np.linalg.norm(W_sft))
    rlvr_norm = float(np.linalg.norm(W_rlvr))
    delta_norm = float(np.linalg.norm(delta))
    ratio = delta_norm / sft_norm if sft_norm > 0 else float("nan")
    dot = float(W_sft.reshape(-1) @ W_rlvr.reshape(-1))
    denom = sft_norm * rlvr_norm
    cosine = dot / denom if denom > 0 else float("nan")

    out = dict(
        name=name,
        category=categorize(name),
        layer=layer_idx(name),
        expert=expert_idx(name),
        shape=list(map(int, W_sft.shape)),
        ndim=int(W_sft.ndim),
        numel=int(W_sft.size),
        sft_norm=sft_norm,
        rlvr_norm=rlvr_norm,
        delta_norm=delta_norm,
        delta_ratio=ratio,
        delta_max_abs=float(np.abs(delta).max()),
        sft_max_abs=float(np.abs(W_sft).max()),
        cosine_similarity=cosine,
    )

    if W_sft.ndim == 2:
        m, n = W_sft.shape
        full = int(min(m, n))
        # For tall-skinny or huge matrices (embed/lm_head: 151936×2048), direct SVD is
        # very slow single-threaded. Instead form the Gram matrix G = ΔW^T ΔW (or ΔW ΔW^T,
        # whichever is smaller), eigendecompose — a *single* BLAS matmul plus cheap eig.
        # This yields ALL singular values via sv = sqrt(eigvalsh(G)).
        GRAM_THRESHOLD = 4096  # use gram-matrix method when max(m,n) > this
        if max(m, n) > GRAM_THRESHOLD:
            # reduce to the smaller dimension's Gram matrix
            if m >= n:
                G = delta.T @ delta  # (n, n)
            else:
                G = delta @ delta.T  # (m, m)
            # eigvalsh returns ascending — flip to descending
            eig = np.linalg.eigvalsh(G.astype(np.float64))[::-1]
            eig = np.clip(eig, 0.0, None)
            sv = np.sqrt(eig)
            gram_method = True
        else:
            sv = np.linalg.svd(delta, compute_uv=False)
            gram_method = False

        r50, r90, r95, r99 = effective_ranks(sv)
        top_for_storage = sv[:TOP_SV]
        sv_count_stored = int(len(sv))
        top1 = float(sv[0]) if len(sv) else 0.0
        stab = stable_rank(sv) if len(sv) else 0.0
        truncated_svd = gram_method

        out.update(
            delta_top_sv=top_for_storage.astype(np.float32).tolist(),
            delta_sv_count=sv_count_stored,
            delta_top1_sv=top1,
            delta_stable_rank=stab,
            eff_rank_50=r50,
            eff_rank_90=r90,
            eff_rank_95=r95,
            eff_rank_99=r99,
            rank_ratio_90=r90 / full,
            rank_ratio_99=r99 / full,
            full_rank=full,
            truncated_svd=truncated_svd,
        )
    else:
        out.update(
            delta_top_sv=None,
            delta_sv_count=None,
            delta_top1_sv=None,
            delta_stable_rank=None,
            eff_rank_50=None,
            eff_rank_90=None,
            eff_rank_95=None,
            eff_rank_99=None,
            rank_ratio_90=None,
            rank_ratio_99=None,
            full_rank=None,
            truncated_svd=None,
        )
    return out


def _worker(args):
    # Ensure single-threaded BLAS inside the worker as well — the module-level env
    # set should have taken effect, but also hard-cap via threadpoolctl in case the
    # child imports numpy from a cached runtime state.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass

    name, sft_path, rlvr_path = args
    with safe_open(sft_path, framework="pt") as f:
        t = f.get_tensor(name)
    W_sft = t.to(dtype=__import__("torch").float32).numpy()
    with safe_open(rlvr_path, framework="pt") as f:
        t = f.get_tensor(name)
    W_rlvr = t.to(dtype=__import__("torch").float32).numpy()
    if W_sft.shape != W_rlvr.shape:
        raise ValueError(f"shape mismatch for {name}: {W_sft.shape} vs {W_rlvr.shape}")
    return analyze_tensor(name, W_sft, W_rlvr)


def build_tensor_index(dir_path):
    idx = Path(dir_path) / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            data = json.load(f)
        wm = data["weight_map"]
        return {k: str(Path(dir_path) / v) for k, v in wm.items()}
    # fallback: scan shards
    d = {}
    for p in sorted(Path(dir_path).glob("model-*.safetensors")):
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                d[k] = str(p)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", default=SFT_DIR)
    ap.add_argument("--rlvr-dir", default=RLVR_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--filter", default=None, help="regex on tensor name; only matching are processed")
    ap.add_argument("--output-name", default="per_tensor_delta")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building tensor indices...", flush=True)
    sft_map = build_tensor_index(args.sft_dir)
    rlvr_map = build_tensor_index(args.rlvr_dir)
    common = sorted(set(sft_map) & set(rlvr_map))
    only_sft = sorted(set(sft_map) - set(rlvr_map))
    only_rlvr = sorted(set(rlvr_map) - set(sft_map))
    print(f"  SFT tensors:   {len(sft_map)}")
    print(f"  RLVR tensors:  {len(rlvr_map)}")
    print(f"  Common:        {len(common)}")
    if only_sft:
        print(f"  Only in SFT:   {len(only_sft)}  sample: {only_sft[:3]}")
    if only_rlvr:
        print(f"  Only in RLVR:  {len(only_rlvr)}  sample: {only_rlvr[:3]}")

    if args.filter:
        pat = re.compile(args.filter)
        common = [n for n in common if pat.search(n)]
        print(f"  After filter /{args.filter}/: {len(common)}")

    tasks = [(n, sft_map[n], rlvr_map[n]) for n in common]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"[{time.strftime('%H:%M:%S')}] Processing {len(tasks)} tensors with {args.workers} workers", flush=True)
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append(dict(name=futures[fut], error=repr(e)))
                print(f"  ERROR {futures[fut]}: {e}", flush=True)
            if i % 200 == 0 or i == len(tasks):
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0.0
                eta = (len(tasks) - i) / rate if rate > 0 else 0.0
                print(
                    f"[{time.strftime('%H:%M:%S')}]   {i}/{len(tasks)} "
                    f"({i / len(tasks) * 100:.1f}%)  {rate:.1f}/s  ETA {eta / 60:.1f}min",
                    flush=True,
                )

    import pandas as pd
    df = pd.DataFrame(results)
    sort_cols = [c for c in ["layer", "category", "expert", "name"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    parquet_path = os.path.join(args.out_dir, args.output_name + ".parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {len(df)} rows → {parquet_path}", flush=True)

    jsonl_path = parquet_path.replace(".parquet", ".jsonl")
    with open(jsonl_path, "w") as f:
        for row in df.to_dict(orient="records"):
            clean = {}
            for k, v in row.items():
                if isinstance(v, float) and (v != v):
                    clean[k] = None
                else:
                    clean[k] = v
            f.write(json.dumps(clean, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
