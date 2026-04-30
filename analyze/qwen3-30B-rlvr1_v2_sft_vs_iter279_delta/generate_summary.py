"""
generate_summary.py

Consumes per_tensor_delta.parquet (output of analyze_weight_delta.py) and produces:

  - layer_summary.md     : per-layer / per-category tables of delta ratio & LoRA rank
  - component_summary.md : aggregated stats by (category × layer)
  - plots/cumulative_energy_sample.png  : cumulative Frobenius energy vs. rank for
                                          a handful of representative tensors
  - plots/ratio_heatmap.png             : delta-ratio heatmap across layers × components
  - plots/rank_heatmap.png              : suggested LoRA-rank (at 90% energy) heatmap
  - plots/rank_ratio_violin.png         : distribution of rank_ratio_90 by category
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = "/shared/dev/shuowei/niletron/rlvr/analyze/qwen3-30B-rlvr1_v2_sft_vs_iter279_delta"


LAYER_CATEGORIES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "router",
    "expert_gate",
    "expert_up",
    "expert_down",
    "input_ln",
    "post_attn_ln",
    "q_norm",
    "k_norm",
]


def agg_by_layer_category(df):
    """Return a DataFrame indexed by layer with columns per category — median delta_ratio."""
    sub = df[df["layer"] >= 0].copy()
    piv = sub.pivot_table(
        index="layer",
        columns="category",
        values="delta_ratio",
        aggfunc="median",
    )
    cols = [c for c in LAYER_CATEGORIES if c in piv.columns]
    return piv[cols]


def agg_rank_by_layer_category(df, which="eff_rank_90"):
    sub = df[(df["layer"] >= 0) & (df["ndim"] == 2)].copy()
    piv = sub.pivot_table(
        index="layer",
        columns="category",
        values=which,
        aggfunc="median",
    )
    cols = [c for c in LAYER_CATEGORIES if c in piv.columns]
    return piv[cols]


def agg_rank_ratio(df, which="rank_ratio_90"):
    sub = df[(df["layer"] >= 0) & (df["ndim"] == 2)].copy()
    piv = sub.pivot_table(
        index="layer",
        columns="category",
        values=which,
        aggfunc="median",
    )
    cols = [c for c in LAYER_CATEGORIES if c in piv.columns]
    return piv[cols]


def write_layer_summary(df, out_path):
    lines = []
    lines.append("# Per-layer weight-delta summary\n")
    lines.append("Columns: median values across all tensors in that (layer, category) cell.\n")
    lines.append("For expert_* categories, the cell aggregates across all 128 experts.\n\n")

    lines.append("## Median delta_ratio  (||ΔW||_F / ||W_sft||_F)\n")
    m = agg_by_layer_category(df)
    lines.append(m.to_markdown(floatfmt=".4f"))
    lines.append("\n\n")

    lines.append("## Median eff_rank_90  (smallest rank capturing 90 % of ΔW Frobenius energy)\n")
    r = agg_rank_by_layer_category(df, "eff_rank_90")
    lines.append(r.to_markdown(floatfmt=".1f"))
    lines.append("\n\n")

    lines.append("## Median rank_ratio_90  (eff_rank_90 / min(m, n))\n")
    rr = agg_rank_ratio(df, "rank_ratio_90")
    lines.append(rr.to_markdown(floatfmt=".3f"))
    lines.append("\n\n")

    lines.append("## Median eff_rank_99\n")
    r99 = agg_rank_by_layer_category(df, "eff_rank_99")
    lines.append(r99.to_markdown(floatfmt=".1f"))
    lines.append("\n\n")

    # non-layer tensors (embed, lm_head, final_norm)
    extra = df[df["layer"] < 0][["name", "category", "shape", "delta_ratio", "cosine_similarity", "eff_rank_90", "eff_rank_99", "full_rank"]]
    if len(extra) > 0:
        lines.append("## Non-layer tensors\n")
        lines.append(extra.to_markdown(index=False, floatfmt=".4g"))
        lines.append("\n")

    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"Wrote {out_path}")


def write_category_summary(df, out_path):
    sub = df[df["ndim"] == 2].copy()
    g = sub.groupby("category").agg(
        n=("name", "count"),
        median_ratio=("delta_ratio", "median"),
        mean_ratio=("delta_ratio", "mean"),
        median_rank_90=("eff_rank_90", "median"),
        median_rank_99=("eff_rank_99", "median"),
        median_full_rank=("full_rank", "median"),
        median_rank_ratio_90=("rank_ratio_90", "median"),
        median_rank_ratio_99=("rank_ratio_99", "median"),
        median_cos=("cosine_similarity", "median"),
        median_stable_rank=("delta_stable_rank", "median"),
    )
    g = g.reindex([c for c in LAYER_CATEGORIES + ["embed", "lm_head"] if c in g.index])
    with open(out_path, "w") as f:
        f.write("# Delta stats aggregated across all layers, by category\n\n")
        f.write(g.to_markdown(floatfmt=".4g"))
        f.write("\n")
    print(f"Wrote {out_path}")


def plot_ratio_heatmap(df, out_path):
    m = agg_by_layer_category(df)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(m.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns, rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index)
    ax.set_xlabel("category")
    ax.set_ylabel("layer")
    ax.set_title("Median delta_ratio  (||ΔW||_F / ||W_sft||_F)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_rank_heatmap(df, out_path):
    r = agg_rank_ratio(df, "rank_ratio_90")
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(r.values, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels(r.columns, rotation=45, ha="right")
    ax.set_yticks(range(r.shape[0]))
    ax.set_yticklabels(r.index)
    ax.set_xlabel("category")
    ax.set_ylabel("layer")
    ax.set_title("Median rank_ratio_90 = eff_rank_90 / min(m,n)\n(fraction of full rank needed to capture 90% of ΔW energy)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_rank_violin(df, out_path):
    sub = df[(df["ndim"] == 2) & df["rank_ratio_90"].notna()].copy()
    cats_present = [c for c in LAYER_CATEGORIES + ["embed", "lm_head"] if c in sub["category"].unique()]
    data = [sub[sub["category"] == c]["rank_ratio_90"].values for c in cats_present]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(cats_present) + 1))
    ax.set_xticklabels(cats_present, rotation=45, ha="right")
    ax.set_ylabel("rank_ratio_90 (eff_rank_90 / min(m,n))")
    ax.set_title("Low-rank-ness of ΔW by category")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cumulative_energy(df, out_path, n_per_cat=2):
    """For each category with stored top singular values, plot cumulative Frobenius energy vs. rank."""
    sub = df[df["delta_top_sv"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    cats_shown = 0
    for cat in LAYER_CATEGORIES + ["embed", "lm_head"]:
        rows = sub[sub["category"] == cat].head(n_per_cat)
        for _, row in rows.iterrows():
            sv = np.array(row["delta_top_sv"], dtype=np.float64)
            if len(sv) == 0:
                continue
            e = sv ** 2
            full_e = row["delta_norm"] ** 2
            cum = np.cumsum(e) / full_e
            ax.plot(range(1, len(cum) + 1), cum, alpha=0.75, label=f"{cat} L{int(row['layer'])}" if cats_shown < 20 else None)
        cats_shown += 1
    ax.set_xlabel("rank k")
    ax.set_ylabel("cumulative Frobenius energy  Σ_{i≤k} σ_i²  /  ||ΔW||_F²")
    ax.set_title("Concentration of ΔW in its top singular directions (top-64 sv)")
    ax.axhline(0.9, color="red", ls="--", alpha=0.5, label="90%")
    ax.axhline(0.99, color="orange", ls="--", alpha=0.5, label="99%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(OUT_DIR, "per_tensor_delta.parquet"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(df["category"].value_counts())

    write_layer_summary(df, os.path.join(args.out_dir, "layer_summary.md"))
    write_category_summary(df, os.path.join(args.out_dir, "category_summary.md"))

    plot_ratio_heatmap(df, os.path.join(args.out_dir, "plots", "ratio_heatmap.png"))
    plot_rank_heatmap(df, os.path.join(args.out_dir, "plots", "rank_heatmap.png"))
    plot_rank_violin(df, os.path.join(args.out_dir, "plots", "rank_ratio_violin.png"))
    plot_cumulative_energy(df, os.path.join(args.out_dir, "plots", "cumulative_energy_sample.png"))


if __name__ == "__main__":
    main()
