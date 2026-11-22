#!/usr/bin/env python3
"""
Concept contribution analysis for TrafficCBM.

Primary output:
1. Ablation-based concept contribution on a chosen split.
2. Exact GA2M main/pair contributions when head="ga2m".

The script is designed for concept-style fusion modes such as:
  - 6+4
  - 6+4|b
  - 6+4+5
  - 6+4+5+3
  - 6+4+3
"""
import argparse
import json
import os
import random
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.traffic_dataset import TrafficDataset, collate_fn, fit_or_load_scalers
from model.encoders.sequence_encoder import SeqConceptEncoder
from model.encoders.stat_groups import GROUP_NAMES as STAT_GROUP_NAMES
from model.traffic_cbm import GA2MHead, build_model
from train.train import (
    _batch_to_kwargs,
    _load_concept_groups,
    _resolve_concept_groups,
    _build_concept_groups,
    _flatten_config,
    _get_stat_info,
    _num_classes,
    _resolve_bytes_branch_flags,
)


SEQ_CONCEPT_NAMES = list(SeqConceptEncoder.GROUP_NAMES)


def _resolve_paths(args) -> Tuple[str, str, str]:
    if args.ckpt_dir:
        ckpt_path = args.ckpt or os.path.join(args.ckpt_dir, "best.pt")
        cfg_path = args.config or os.path.join(args.ckpt_dir, "config.json")
        output_dir = args.output_dir or args.ckpt_dir
    else:
        if not args.ckpt or not args.config:
            raise ValueError("--ckpt_dir or (--ckpt and --config) is required")
        ckpt_path = args.ckpt
        cfg_path = args.config
        output_dir = args.output_dir or os.path.dirname(ckpt_path)
    return ckpt_path, cfg_path, output_dir


def _load_cfg(cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return _flatten_config(json.load(f))


def _concept_names(
    fusion_mode: str,
    num_packets: int,
    num_groups: int,
    use_stat: bool = True,
    use_seq: bool = True,
    use_bytes: bool = True,
    use_bytes_packet: bool = True,
    use_bytes_cross: bool = True,
) -> List[str]:
    stat_names = [f"stat/{name}" for name in STAT_GROUP_NAMES]
    seq_names = [f"seq/{name}" for name in SEQ_CONCEPT_NAMES]
    stat_part = stat_names if use_stat else []
    seq_part = seq_names if use_seq else []

    if fusion_mode == "6+4":
        return stat_part + seq_part
    if fusion_mode == "6+4|b":
        return stat_part + seq_part
    if fusion_mode == "6+4+5":
        bytes_part = [f"bytes/pkt_{i+1}" for i in range(num_packets)] if (use_bytes and use_bytes_packet) else []
        return stat_part + seq_part + bytes_part

    if num_groups == 3:
        cross_names = ["bytes/cross_front", "bytes/cross_middle", "bytes/cross_back"]
    else:
        cross_names = [f"bytes/cross_{i+1}" for i in range(num_groups)]

    if fusion_mode == "6+4+5+3":
        bytes_part = []
        if use_bytes and use_bytes_packet:
            bytes_part.extend([f"bytes/pkt_{i+1}" for i in range(num_packets)])
        if use_bytes and use_bytes_cross:
            bytes_part.extend(cross_names)
        return stat_part + seq_part + bytes_part
    if fusion_mode == "6+4+3":
        bytes_part = cross_names if (use_bytes and use_bytes_cross) else []
        return stat_part + seq_part + bytes_part
    if fusion_mode == "6/6/6":
        n = len(stat_names)
        return [f"shared/concept_{i+1}" for i in range(n)]
    raise ValueError(f"Unsupported fusion_mode for concept analysis: {fusion_mode}")


def _fused_key(extras: Dict[str, torch.Tensor]) -> str:
    for key in ("c_final", "c_fused", "fused"):
        if key in extras:
            return key
    raise RuntimeError("Model extras does not contain c_final/c_fused/fused")


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_concepts = []

    for batch in loader:
        y = batch["label"].to(device)
        kwargs = _batch_to_kwargs(batch, device)
        logits, extras = model(**kwargs)
        fused = extras[_fused_key(extras)]
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_concepts.append(fused.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    concepts = torch.cat(all_concepts, dim=0)
    return logits, labels, concepts


def _head_forward(model, fused: torch.Tensor) -> torch.Tensor:
    return model.head(fused)[0] if getattr(model, "_use_taylor", False) else model.head(fused)


@torch.no_grad()
def ablation_contrib(model, concepts: torch.Tensor, target_idx: torch.Tensor, baseline: torch.Tensor):
    full_logits = _head_forward(model, concepts)
    full_target = full_logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    deltas = []
    for k in range(concepts.size(1)):
        mod = concepts.clone()
        mod[:, k] = baseline[k]
        ablated_logits = _head_forward(model, mod)
        ablated_target = ablated_logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        deltas.append((full_target - ablated_target).detach().cpu())
    return torch.stack(deltas, dim=1)


@torch.no_grad()
def ga2m_exact_contrib(model, concepts: torch.Tensor, target_idx: torch.Tensor):
    if not isinstance(model.head, GA2MHead):
        return None

    head = model.head
    main_vals = []
    for i, effect in enumerate(head.main_effects):
        out = effect(concepts[:, i : i + 1])
        main_vals.append(out.gather(1, target_idx.unsqueeze(1)).squeeze(1).detach().cpu())
    main_vals = torch.stack(main_vals, dim=1)

    pair_vals = []
    pair_names = []
    for (i, j), effect in zip(head.pair_indices, head.pair_effects):
        out = effect(concepts[:, [i, j]])
        pair_vals.append(out.gather(1, target_idx.unsqueeze(1)).squeeze(1).detach().cpu())
        pair_names.append((i, j))
    pair_vals = torch.stack(pair_vals, dim=1) if pair_vals else torch.empty((concepts.size(0), 0))
    return main_vals, pair_vals, pair_names


def _write_ablation_report(
    out_path: str,
    concept_names: List[str],
    deltas: torch.Tensor,
    labels: torch.Tensor,
    id2label: Dict[int, str],
):
    lines = []
    lines.append("=" * 88)
    lines.append("Concept Contribution Analysis (Ablation)")
    lines.append("=" * 88)
    lines.append(f"samples={labels.numel()}  concepts={len(concept_names)}")
    lines.append("")
    lines.append(f"{'rank':>4}  {'concept':<28}  {'mean_delta':>11}  {'mean_abs_delta':>14}")
    lines.append("-" * 88)

    mean_delta = deltas.mean(dim=0).numpy()
    mean_abs_delta = deltas.abs().mean(dim=0).numpy()
    order = np.argsort(-mean_abs_delta)
    for rank, idx in enumerate(order, 1):
        lines.append(
            f"{rank:>4}  {concept_names[idx]:<28}  {mean_delta[idx]:>11.4f}  {mean_abs_delta[idx]:>14.4f}"
        )

    lines.append("")
    lines.append("Per-class mean_abs_delta")
    lines.append("")
    header = f"{'class':<24}  " + "  ".join(f"{i+1:>6}" for i in range(len(concept_names)))
    lines.append(header)
    lines.append("-" * len(header))
    for c in sorted(torch.unique(labels).tolist()):
        mask = labels == c
        vals = deltas[mask].abs().mean(dim=0).numpy()
        cname = id2label.get(int(c), str(c))
        lines.append(f"{cname[:24]:<24}  " + "  ".join(f"{v:>6.3f}" for v in vals))

    report = "\n".join(lines)
    print(report, flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[save] ablation report -> {out_path}", flush=True)


def _write_ga2m_report(
    out_path: str,
    concept_names: List[str],
    main_vals: torch.Tensor,
    pair_vals: torch.Tensor,
    pair_names: List[Tuple[int, int]],
):
    lines = []
    lines.append("=" * 88)
    lines.append("GA2M Exact Contribution Analysis")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Main effects (ranked by mean absolute contribution)")
    lines.append("")
    main_abs = main_vals.abs().mean(dim=0).numpy()
    main_mean = main_vals.mean(dim=0).numpy()
    order = np.argsort(-main_abs)
    lines.append(f"{'rank':>4}  {'concept':<28}  {'mean':>10}  {'mean_abs':>10}")
    lines.append("-" * 88)
    for rank, idx in enumerate(order, 1):
        lines.append(f"{rank:>4}  {concept_names[idx]:<28}  {main_mean[idx]:>10.4f}  {main_abs[idx]:>10.4f}")

    if pair_vals.numel() > 0:
        lines.append("")
        lines.append("Top pair interactions (ranked by mean absolute contribution)")
        lines.append("")
        pair_abs = pair_vals.abs().mean(dim=0).numpy()
        pair_mean = pair_vals.mean(dim=0).numpy()
        pair_order = np.argsort(-pair_abs)[:30]
        lines.append(f"{'rank':>4}  {'pair':<46}  {'mean':>10}  {'mean_abs':>10}")
        lines.append("-" * 88)
        for rank, idx in enumerate(pair_order, 1):
            i, j = pair_names[idx]
            pname = f"{concept_names[i]} x {concept_names[j]}"
            lines.append(f"{rank:>4}  {pname:<46}  {pair_mean[idx]:>10.4f}  {pair_abs[idx]:>10.4f}")

    report = "\n".join(lines)
    print(report, flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[save] ga2m report -> {out_path}", flush=True)


def _load_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        available = set(getattr(plt.style, "available", []))
        for style_name in (
            "seaborn-v0_8-whitegrid",
            "seaborn-whitegrid",
            "ggplot",
            "default",
        ):
            if style_name in available or style_name == "default":
                plt.style.use(style_name)
                break
        # Match the typography and axes used in the paper figures.
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 9.5,
                "legend.fontsize": 8.8,
                "lines.linewidth": 1.0,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "text.usetex": False,
                "axes.linewidth": 0.85,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "figure.dpi": 160,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.alpha": 0.18,
                "grid.linewidth": 0.65,
            }
        )
        return plt
    except ImportError:
        print("[warn] matplotlib not found, skipping plots", flush=True)
        return None
    except Exception as e:
        print(f"[warn] matplotlib plotting setup failed: {e}", flush=True)
        return None


def _short_names(concept_names: List[str]) -> List[str]:
    return [name.split("/", 1)[-1] for name in concept_names]


def _concept_family(name: str) -> str:
    if name.startswith("stat/"):
        return "stat"
    if name.startswith("seq/"):
        return "seq"
    if name.startswith("bytes/"):
        return "bytes"
    return "other"


def _paper_label(name: str) -> str:
    suffix = name.split("/", 1)[-1] if "/" in name else name
    suffix = suffix.replace("packet_", "pkt ")
    suffix = suffix.replace("cross_", "cross ")
    suffix = suffix.replace("_", " ")
    return suffix


def _wrap_label(name: str, width: int = 22) -> str:
    return textwrap.fill(_paper_label(name), width=width)


def _safe_plot_name(name: str) -> str:
    text = _paper_label(name).lower().replace(" × ", "_x_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "concept"


def _mean_and_ci(vals: torch.Tensor, use_abs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    x = vals.abs() if use_abs else vals
    mean = x.mean(dim=0)
    if x.size(0) <= 1:
        ci = torch.zeros_like(mean)
    else:
        std = x.std(dim=0, unbiased=False)
        sem = std / np.sqrt(float(x.size(0)))
        ci = 1.96 * sem
    return mean.detach().cpu().numpy(), ci.detach().cpu().numpy()


def _family_colors() -> Dict[str, str]:
    return {
        "stat": "#4C78A8",
        "seq": "#59A14F",
        "bytes": "#E15759",
        "other": "#9C9C9C",
    }


def _barh_plot(
    out_path: str,
    names: List[str],
    vals: np.ndarray,
    errs: np.ndarray,
    xlabel: str,
):
    plt = _load_plt()
    if plt is None:
        return

    from matplotlib.patches import Patch

    family_colors = {
        "stat": "#4C78A8",
        "seq": "#59A14F",
        "bytes": "#E15759",
        "other": "#9C9C9C",
    }
    colors = [family_colors.get(_concept_family(name), "#9C9C9C") for name in names]
    wrapped_names = [_wrap_label(name, width=24) for name in names]

    fig_h = max(7.6, len(names) * 0.52)
    fig, ax = plt.subplots(figsize=(10.6, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(names))

    bars = ax.barh(
        y,
        vals,
        xerr=errs,
        color=colors,
        alpha=0.92,
        edgecolor="none",
        linewidth=0.0,
        height=0.66,
        error_kw={
            "ecolor": "#525252",
            "elinewidth": 1.2,
            "capsize": 2.5,
            "capthick": 1.2,
            "alpha": 1.0,
        },
    )

    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_names, fontsize=12)
    for tick in ax.get_yticklabels():
        tick.set_color("#303030")
        tick.set_fontweight("medium")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(axis="x", linestyle=(0, (2, 3)), color="#DEDEDE", linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", pad=10)
    ax.tick_params(axis="x", labelsize=12, length=4, width=0.9, color="#202020", direction="out")
    ax.tick_params(axis="y", length=3, width=0.9, color="#202020", direction="out")
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#202020")
        ax.spines[spine].set_linewidth(1.0)

    xmax = float(np.max(vals + errs)) if len(vals) else 1.0
    xmax = max(xmax, 1e-6)
    right_pad = max(0.14 * xmax, 0.08)
    ax.set_xlim(0.0, xmax + right_pad)

    text_pad = 0.012 * (xmax + right_pad)
    for bar, v, e in zip(bars, vals, errs):
        ymid = bar.get_y() + bar.get_height() / 2.0
        label = f"{v:.3f}"
        x = min(v + e + text_pad, xmax + right_pad - text_pad)
        ax.text(x, ymid, label, va="center", ha="left", fontsize=10.5, color="#2F2F2F")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] plot -> {out_path}", flush=True)


def _signed_barh_plot(
    out_path: str,
    names: List[str],
    vals: np.ndarray,
    abs_vals: np.ndarray,
    errs: np.ndarray,
    xlabel: str,
):
    plt = _load_plt()
    if plt is None:
        return

    from matplotlib.patches import Patch

    family_colors = {
        "stat": "#4C78A8",
        "seq": "#59A14F",
        "bytes": "#E15759",
        "other": "#9C9C9C",
    }
    colors = [family_colors.get(_concept_family(name), "#9C9C9C") for name in names]
    wrapped_names = [_wrap_label(name, width=24) for name in names]

    fig_h = max(7.6, len(names) * 0.52)
    fig, ax = plt.subplots(figsize=(10.6, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(names))

    # Autoscale around the signed mean effect so the main signal stays readable.
    vmax_main = float(np.max(np.abs(vals) + errs)) if len(vals) else 1.0
    vmax_main = max(vmax_main, 1e-6)
    pad = max(0.14 * vmax_main, 0.05)
    span = vmax_main + pad
    # Extra x-range so end-of-bar value labels stay inside the axes.
    label_margin = max(0.20 * span, 0.07)
    x_right = span + label_margin
    x_left = -span - label_margin

    # Keep a gray reference for mean|effect|, but scale it relatively so we
    # preserve the ranking cue without forcing a second numeric axis.
    abs_ref = np.zeros_like(abs_vals)
    if len(abs_vals):
        abs_ref = abs_vals / max(float(np.max(abs_vals)), 1e-6) * (0.92 * span)

    # A thin centered gray reference band shows mean |effect| without dominating the chart.
    ax.barh(
        y,
        2.0 * abs_ref,
        left=-abs_ref,
        color="#D0D0D0",
        alpha=0.55,
        edgecolor="none",
        linewidth=0.0,
        height=0.22,
        zorder=1,
    )

    bars = ax.barh(
        y,
        vals,
        xerr=errs,
        color=colors,
        alpha=0.92,
        edgecolor="none",
        linewidth=0.0,
        height=0.52,
        error_kw={
            "ecolor": "#525252",
            "elinewidth": 1.2,
            "capsize": 2.5,
            "capthick": 1.2,
            "alpha": 1.0,
        },
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_names, fontsize=16)
    for tick in ax.get_yticklabels():
        tick.set_color("#303030")
        tick.set_fontweight("medium")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=16, fontweight="semibold", labelpad=3.5)
    ax.set_ylabel("Concepts", fontsize=16, fontweight="semibold", labelpad=5)
    ax.grid(axis="x", linestyle=(0, (2, 3)), color="#DEDEDE", linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", pad=10)
    ax.tick_params(axis="x", labelsize=16, length=4, width=0.9, color="#202020", direction="out")
    ax.tick_params(axis="y", length=3, width=0.9, color="#202020", direction="out")
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#202020")
        ax.spines[spine].set_linewidth(1.0)
    ax.axvline(0.0, color="#7F7F7F", linewidth=1.1, zorder=2)
    ax.set_xlim(x_left, x_right)

    text_pad = 0.02 * span
    # Inset from axis edge so ha=left / ha=right numeric labels (clip_on=True) stay visible.
    text_inset = max(0.08 * span, 0.04)
    for bar, v, e in zip(bars, vals, errs):
        ymid = bar.get_y() + bar.get_height() / 2.0
        label = f"{v:.3f}"
        if v >= 0:
            x = min(v + e + text_pad, x_right - text_inset)
            ha = "left"
            color = "#2F2F2F"
        else:
            x = max(v - e - text_pad, x_left + text_inset)
            ha = "right"
            color = "#2F2F2F"
        ax.text(x, ymid, label, va="center", ha=ha, fontsize=14, color=color)

    legend_handles = [
        Patch(facecolor="#D0D0D0", edgecolor="none", alpha=0.55, label="Abs. effect"),
        Patch(facecolor=family_colors["stat"], edgecolor="none", label="Stat"),
        Patch(facecolor=family_colors["seq"], edgecolor="none", label="Seq"),
        Patch(facecolor=family_colors["bytes"], edgecolor="none", label="Bytes"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.995, 0.998),
        borderaxespad=0.25,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=13,
    )

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[save] plot -> {out_path}", flush=True)


def plot_ablation_bar(out_path: str, concept_names: List[str], deltas: torch.Tensor):
    mean_abs, ci_abs = _mean_and_ci(deltas, use_abs=True)
    order = np.argsort(-mean_abs)
    names = [concept_names[i] for i in order]
    vals = mean_abs[order]
    errs = ci_abs[order]
    _barh_plot(
        out_path=out_path,
        names=names,
        vals=vals,
        errs=errs,
        xlabel="Mean absolute change in target logit (95% CI)",
    )


def plot_ga2m_main_bar(
    out_path: str,
    concept_names: List[str],
    main_vals: torch.Tensor,
    top_k: int = 15,
):
    mean_abs, _ = _mean_and_ci(main_vals, use_abs=True)
    mean_signed, ci_signed = _mean_and_ci(main_vals, use_abs=False)
    order = np.argsort(-mean_abs)[:top_k]
    names = [concept_names[i] for i in order]
    abs_vals = mean_abs[order]
    signed = mean_signed[order]
    errs = ci_signed[order]
    _signed_barh_plot(
        out_path=out_path,
        names=names,
        vals=signed,
        abs_vals=abs_vals,
        errs=errs,
        xlabel="Mean main effect",
    )


def plot_ga2m_term_ranking(
    out_path: str,
    concept_names: List[str],
    main_vals: torch.Tensor,
    pair_vals: torch.Tensor,
    pair_names: List[Tuple[int, int]],
    top_first_k: int = 5,
    top_second_k: int = 10,
):
    plt = _load_plt()
    if plt is None:
        return

    from matplotlib.patches import Patch

    main_abs = main_vals.abs().mean(dim=0).numpy()
    first_rows = []
    for idx, val in enumerate(main_abs):
        first_rows.append({
            "label": _paper_label(concept_names[idx]),
            "value": float(val),
            "kind": "first",
        })
    first_rows = sorted(first_rows, key=lambda x: x["value"], reverse=True)[:top_first_k]

    second_rows = []
    if pair_vals.numel() > 0:
        pair_abs = pair_vals.abs().mean(dim=0).numpy()
        for idx, val in enumerate(pair_abs):
            i, j = pair_names[idx]
            second_rows.append({
                "label": f"{_paper_label(concept_names[i])} × {_paper_label(concept_names[j])}",
                "value": float(val),
                "kind": "second",
            })
        second_rows = sorted(second_rows, key=lambda x: x["value"], reverse=True)[:top_second_k]

    term_rows = first_rows + second_rows
    if not term_rows:
        return

    labels = [textwrap.fill(row["label"], width=24) for row in term_rows]
    vals = np.asarray([row["value"] for row in term_rows], dtype=np.float32)
    colors = ["#5B86B0" if row["kind"] == "first" else "#8E8E8E" for row in term_rows]

    fig_h = max(6.0, len(term_rows) * 0.42)
    fig, ax = plt.subplots(figsize=(10.4, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(term_rows))
    bars = ax.barh(
        y,
        vals,
        color=colors,
        alpha=0.96,
        edgecolor="none",
        linewidth=0.0,
        height=0.68,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    for tick in ax.get_yticklabels():
        tick.set_color("#2F2F2F")
        tick.set_fontstyle("italic")
    ax.invert_yaxis()

    ax.set_xlabel("Mean absolute effect", fontsize=16, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Concepts", fontsize=16, fontweight="semibold", labelpad=6)
    ax.grid(axis="x", linestyle=(0, (2, 3)), color="#DEDEDE", linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", pad=8, length=3, width=0.9, color="#202020", direction="out")
    ax.tick_params(axis="x", labelsize=15, length=4, width=0.9, color="#202020", direction="out")
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#202020")
        ax.spines[spine].set_linewidth(1.0)

    if first_rows and second_rows:
        ax.axhline(len(first_rows) - 0.5, color="#B8B8B8", linewidth=1.0, linestyle=(0, (3, 3)))

    xmax = float(np.max(vals)) if len(vals) else 1.0
    xmax = max(xmax, 1e-6)
    right_pad = max(0.14 * xmax, 0.08)
    x_hi = xmax + right_pad
    ax.set_xlim(0.0, x_hi)

    text_pad = 0.012 * x_hi
    text_inset = max(0.045 * xmax, 0.03)
    for bar, row in zip(bars, term_rows):
        ymid = bar.get_y() + bar.get_height() / 2.0
        x = min(row["value"] + text_pad, x_hi - text_inset)
        ax.text(x, ymid, f"{row['value']:.3f}", va="center", ha="left", fontsize=13, color="#2F2F2F")

    ax.legend(
        handles=[
            Patch(facecolor="#5B86B0", edgecolor="none", label="First Order"),
            Patch(facecolor="#8E8E8E", edgecolor="none", label="Second Order"),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=14,
    )

    ax.margins(y=0.02)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.06, hspace=0.0, wspace=0.0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)
    print(f"[save] ga2m term ranking -> {out_path}", flush=True)


def _smooth_main_effect_curve(
    curve_x: List[float],
    curve_y: List[float],
    n_fine: int = 220,
    window: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate binned means and smooth them with a moving average."""
    x = np.asarray(curve_x, dtype=np.float64)
    y = np.asarray(curve_y, dtype=np.float64)
    if x.size < 2:
        return x, y
    o = np.argsort(x)
    x, y = x[o], y[o]
    xf = np.linspace(x[0], x[-1], max(n_fine, x.size * 4))
    yi = np.interp(xf, x, y)
    w = int(min(max(3, window), max(3, yi.size // 3 * 2 + 1)))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yi_pad = np.pad(yi, (pad, pad), mode="edge")
    yi_s = np.convolve(yi_pad, np.ones(w) / w, mode="valid")
    return xf, yi_s


def _draw_ga2m_shape_axis(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
    density_bins: int = 25,
    line_bins: int = 100,
):
    line_color = "#5B6EC7"
    step_color = "#C8D0EE"
    density_color = "#D94B5B"

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        ax.set_visible(False)
        return

    q_low, q_high = np.quantile(x, [0.05, 0.95]) if x.size >= 8 else (x.min(), x.max())
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
        q_low, q_high = float(x.min()), float(x.max())
    span = q_high - q_low
    if span <= 0:
        span = max(abs(q_low), 1.0) * 0.1 + 1e-3
    x_low = q_low
    x_high = q_high if q_high > q_low else q_low + span

    mask = (x >= x_low) & (x <= x_high)
    x_plot = x[mask]
    y_plot = y[mask]
    if x_plot.size < 4:
        x_plot = x
        y_plot = y

    x_min, x_max = float(x_plot.min()), float(x_plot.max())
    if x_min == x_max:
        x_max = x_min + 1e-6

    # Show coarse bins as background bands.
    edges_density = np.linspace(x_min, x_max, density_bins + 1)
    hist, _ = np.histogram(x_plot, bins=edges_density, density=True)
    hist = hist.astype(np.float32)
    hist_max = float(hist.max()) if hist.size else 0.0

    y_abs = max(float(np.max(np.abs(y_plot))), 1e-6)
    y_lim = max(0.48, 1.18 * y_abs)
    ax.set_ylim(-y_lim, y_lim)

    for left, right, h in zip(edges_density[:-1], edges_density[1:], hist):
        if hist_max <= 0:
            alpha = 0.0
        else:
            alpha = 0.05 + 0.45 * float(h / hist_max)
        ax.axvspan(left, right, ymin=0.0, ymax=1.0, color=density_color, alpha=alpha, ec="none", zorder=0)

    # Show finer bins as a line.
    edges_line = np.linspace(x_min, x_max, line_bins + 1)
    centers_line = 0.5 * (edges_line[:-1] + edges_line[1:])
    bin_ids = np.digitize(x_plot, edges_line[1:-1], right=False)
    curve_x = []
    curve_y = []
    for b_idx, center in enumerate(centers_line):
        sel = bin_ids == b_idx
        if np.any(sel):
            curve_x.append(center)
            curve_y.append(float(np.mean(y_plot[sel])))

    if len(curve_x) >= 2:
        ax.step(
            curve_x,
            curve_y,
            where="mid",
            color=step_color,
            linewidth=1.6,
            zorder=3,
        )
        xs, ys = _smooth_main_effect_curve(curve_x, curve_y)
        ax.plot(xs, ys, color=line_color, linewidth=2.0, zorder=4, clip_on=True)
    else:
        sort_idx = np.argsort(x_plot)
        xp = x_plot[sort_idx]
        yp = y_plot[sort_idx]
        if xp.size >= 2:
            ax.plot(xp, yp, color=step_color, linewidth=1.3, zorder=3)
            xs, ys = _smooth_main_effect_curve(list(xp), list(yp))
            ax.plot(xs, ys, color=line_color, linewidth=2.0, zorder=4, clip_on=True)
        else:
            ax.plot(xp, yp, color=line_color, linewidth=1.3, zorder=3)

    ax.axhline(0.0, color="#9A9A9A", linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    ax.set_xlim(x_low, x_high)
    ax.margins(x=0)
    ax.set_title(title, fontsize=11.5, pad=4.5, fontweight="semibold")
    ax.grid(axis="y", linestyle=(0, (2, 3)), color="#E2E2E2", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8.5, length=3, width=0.8, color="#202020", direction="out")
    ax.tick_params(axis="y", labelsize=8.5, length=3, width=0.8, color="#202020", direction="out")
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#202020")
        ax.spines[spine].set_linewidth(0.9)

    if show_ylabel:
        ax.set_ylabel("Main effect", fontsize=9.5)
    if show_xlabel:
        ax.set_xlabel("Concept value", fontsize=9.5)


def plot_ga2m_shape_panel(
    out_path: str,
    concept_names: List[str],
    concepts: torch.Tensor,
    main_vals: torch.Tensor,
    top_k: Optional[int] = None,
    density_bins: int = 25,
    line_bins: int = 100,
):
    plt = _load_plt()
    if plt is None or main_vals.numel() == 0:
        return

    concepts_np = concepts.detach().cpu().numpy()
    main_np = main_vals.detach().cpu().numpy()
    mean_abs = np.abs(main_np).mean(axis=0)
    order = np.argsort(-mean_abs)
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    n_show = len(order)
    if n_show == 0:
        return

    ncols = 3 if n_show > 3 else n_show
    nrows = int(np.ceil(n_show / max(ncols, 1)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.15 * ncols, 1.85 * nrows + 0.1),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for panel_idx, concept_idx in enumerate(order):
        r = panel_idx // ncols
        c = panel_idx % ncols
        ax = axes[r, c]
        _draw_ga2m_shape_axis(
            ax=ax,
            x=concepts_np[:, concept_idx],
            y=main_np[:, concept_idx],
            title=_paper_label(concept_names[concept_idx]).title(),
            show_xlabel=(r == nrows - 1),
            show_ylabel=(c == 0),
            density_bins=density_bins,
            line_bins=line_bins,
        )

    for panel_idx in range(n_show, nrows * ncols):
        r = panel_idx // ncols
        c = panel_idx % ncols
        axes[r, c].set_visible(False)

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[save] ga2m shape panel -> {out_path}", flush=True)


# Plot selected shape functions in a fixed order.
SELECTED_SHAPE_PAPER_LABELS = ["timing", "volume", "pkt time", "flag count", "cross front", "pkt 1"]


def plot_ga2m_shape_panel_selected(
    out_path: str,
    concept_names: List[str],
    concepts: torch.Tensor,
    main_vals: torch.Tensor,
    selected_labels: Optional[List[str]] = None,
    density_bins: int = 25,
    line_bins: int = 100,
):
    """Plot shape functions in the order of selected_labels."""
    plt = _load_plt()
    if plt is None or main_vals.numel() == 0:
        return

    labels = selected_labels or SELECTED_SHAPE_PAPER_LABELS

    order = []
    for lb in labels:
        lb_norm = lb.lower().strip()
        for idx, name in enumerate(concept_names):
            if _paper_label(name).lower().strip() == lb_norm:
                order.append(idx)
                break

    if not order:
        return

    concepts_np = concepts.detach().cpu().numpy()
    main_np = main_vals.detach().cpu().numpy()
    n_show = len(order)
    ncols = 3 if n_show > 3 else n_show
    nrows = int(np.ceil(n_show / max(ncols, 1)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.15 * ncols, 1.85 * nrows + 0.1),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for panel_idx, concept_idx in enumerate(order):
        r = panel_idx // ncols
        c = panel_idx % ncols
        ax = axes[r, c]
        _draw_ga2m_shape_axis(
            ax=ax,
            x=concepts_np[:, concept_idx],
            y=main_np[:, concept_idx],
            title=_paper_label(concept_names[concept_idx]).title(),
            show_xlabel=(r == nrows - 1),
            show_ylabel=(c == 0),
            density_bins=density_bins,
            line_bins=line_bins,
        )

    for panel_idx in range(n_show, nrows * ncols):
        r = panel_idx // ncols
        c = panel_idx % ncols
        axes[r, c].set_visible(False)

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[save] ga2m shape panel (selected) -> {out_path}", flush=True)


def plot_ga2m_shape_individuals(
    out_dir: str,
    concept_names: List[str],
    concepts: torch.Tensor,
    main_vals: torch.Tensor,
    density_bins: int = 25,
    line_bins: int = 100,
):
    plt = _load_plt()
    if plt is None or main_vals.numel() == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    concepts_np = concepts.detach().cpu().numpy()
    main_np = main_vals.detach().cpu().numpy()
    mean_abs = np.abs(main_np).mean(axis=0)
    order = np.argsort(-mean_abs)

    for rank, concept_idx in enumerate(order, 1):
        fig, ax = plt.subplots(figsize=(5.2, 2.0), constrained_layout=True)
        _draw_ga2m_shape_axis(
            ax=ax,
            x=concepts_np[:, concept_idx],
            y=main_np[:, concept_idx],
            title=_paper_label(concept_names[concept_idx]).title(),
            show_xlabel=True,
            show_ylabel=True,
            density_bins=density_bins,
            line_bins=line_bins,
        )
        fname = f"{rank:02d}_{_safe_plot_name(concept_names[concept_idx])}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(f"[save] ga2m shape individuals -> {out_dir}", flush=True)


def plot_bar_panel(
    out_path: str,
    concept_names: List[str],
    deltas: torch.Tensor,
    main_vals: torch.Tensor,
):
    plt = _load_plt()
    if plt is None:
        return

    from matplotlib.patches import Patch

    family_colors = _family_colors()

    abl_mean_abs, abl_ci = _mean_and_ci(deltas, use_abs=True)
    abl_order = np.argsort(-abl_mean_abs)
    abl_names = [concept_names[i] for i in abl_order]
    abl_vals = abl_mean_abs[abl_order]
    abl_errs = abl_ci[abl_order]
    abl_colors = [family_colors.get(_concept_family(name), "#9C9C9C") for name in abl_names]
    abl_labels = [_wrap_label(name, width=16) for name in abl_names]

    main_mean_abs, _ = _mean_and_ci(main_vals, use_abs=True)
    main_mean, main_ci = _mean_and_ci(main_vals, use_abs=False)
    main_order = np.argsort(-main_mean_abs)
    main_names = [concept_names[i] for i in main_order]
    main_vals_sorted = main_mean[main_order]
    main_abs_sorted = main_mean_abs[main_order]
    main_errs = main_ci[main_order]
    main_colors = [family_colors.get(_concept_family(name), "#9C9C9C") for name in main_names]
    main_labels = [_wrap_label(name, width=16) for name in main_names]

    fig, axes = plt.subplots(
        1, 2, figsize=(15.6, 7.4), gridspec_kw={"wspace": 0.16}, constrained_layout=True
    )
    fig.patch.set_facecolor("white")

    def _style_axes(ax):
        ax.set_facecolor("white")
        ax.grid(axis="x", linestyle=(0, (2, 3)), color="#DEDEDE", linewidth=0.75)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", pad=8, length=3, width=0.9, color="#202020", direction="out")
        ax.tick_params(axis="x", labelsize=10.5, length=4, width=0.9, color="#202020", direction="out")
        for spine in ("left", "right", "top", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#202020")
            ax.spines[spine].set_linewidth(1.0)

    # Left: ablation
    ax = axes[0]
    y = np.arange(len(abl_names))
    bars = ax.barh(
        y,
        abl_vals,
        xerr=abl_errs,
        color=abl_colors,
        alpha=0.92,
        edgecolor="none",
        linewidth=0.0,
        height=0.62,
        error_kw={
            "ecolor": "#525252",
            "elinewidth": 1.1,
            "capsize": 2.2,
            "capthick": 1.1,
            "alpha": 1.0,
        },
    )
    ax.set_yticks(y)
    ax.set_yticklabels(abl_labels, fontsize=10.5)
    for tick in ax.get_yticklabels():
        tick.set_color("#303030")
        tick.set_fontweight("medium")
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute change in target logit", fontsize=11)
    _style_axes(ax)
    xmax = float(np.max(abl_vals + abl_errs)) if len(abl_vals) else 1.0
    xmax = max(xmax, 1e-6)
    right_pad = max(0.10 * xmax, 0.05)
    ax.set_xlim(0.0, xmax + right_pad)
    text_pad = 0.010 * (xmax + right_pad)
    for bar, v, e in zip(bars, abl_vals, abl_errs):
        ymid = bar.get_y() + bar.get_height() / 2.0
        x = min(v + e + text_pad, xmax + right_pad - text_pad)
        ax.text(x, ymid, f"{v:.3f}", va="center", ha="left", fontsize=9.2, color="#2F2F2F")

    # Right: GA2M main
    ax = axes[1]
    y = np.arange(len(main_names))
    vmax_main = float(np.max(np.abs(main_vals_sorted) + main_errs)) if len(main_vals_sorted) else 1.0
    vmax_main = max(vmax_main, 1e-6)
    pad = max(0.10 * vmax_main, 0.04)
    span = vmax_main + pad
    abs_ref = np.zeros_like(main_abs_sorted)
    if len(main_abs_sorted):
        abs_ref = main_abs_sorted / max(float(np.max(main_abs_sorted)), 1e-6) * (0.92 * span)
    ax.barh(
        y,
        2.0 * abs_ref,
        left=-abs_ref,
        color="#D0D0D0",
        alpha=0.55,
        edgecolor="none",
        linewidth=0.0,
        height=0.26,
        zorder=1,
    )
    bars = ax.barh(
        y,
        main_vals_sorted,
        xerr=main_errs,
        color=main_colors,
        alpha=0.92,
        edgecolor="none",
        linewidth=0.0,
        height=0.50,
        error_kw={
            "ecolor": "#525252",
            "elinewidth": 1.1,
            "capsize": 2.2,
            "capthick": 1.1,
            "alpha": 1.0,
        },
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(main_labels, fontsize=10.5)
    for tick in ax.get_yticklabels():
        tick.set_color("#303030")
        tick.set_fontweight("medium")
    ax.invert_yaxis()
    ax.set_xlabel("Mean main effect", fontsize=11)
    _style_axes(ax)
    ax.axvline(0.0, color="#7F7F7F", linewidth=1.1, zorder=2)
    ax.set_xlim(-span, span)
    text_pad = 0.018 * span
    for bar, v, e in zip(bars, main_vals_sorted, main_errs):
        ymid = bar.get_y() + bar.get_height() / 2.0
        if v >= 0:
            x = min(v + e + text_pad, span - text_pad)
            ha = "left"
        else:
            x = max(v - e - text_pad, -span + text_pad)
            ha = "right"
        ax.text(x, ymid, f"{v:.3f}", va="center", ha=ha, fontsize=9.2, color="#2F2F2F")
    legend_handles = [
        Patch(facecolor="#D0D0D0", edgecolor="none", alpha=0.55, label="Abs. effect"),
        Patch(facecolor=family_colors["stat"], edgecolor="none", label="Stat"),
        Patch(facecolor=family_colors["seq"], edgecolor="none", label="Seq"),
        Patch(facecolor=family_colors["bytes"], edgecolor="none", label="Bytes"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.995, 0.998),
        borderaxespad=0.2,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
        fontsize=8.4,
    )
    ax.text(
        0.0,
        -0.10,
        "Colored bars indicate net mean effect. Features are sorted by mean absolute effect magnitude.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#666666",
    )

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] panel plot -> {out_path}", flush=True)


def plot_ga2m_pair_heatmap(
    out_path: str,
    concept_names: List[str],
    pair_vals: torch.Tensor,
    pair_names: List[Tuple[int, int]],
):
    plt = _load_plt()
    if plt is None or pair_vals.numel() == 0:
        return

    n = len(concept_names)
    mat = np.zeros((n, n), dtype=np.float32)
    pair_abs = pair_vals.abs().mean(dim=0).numpy()
    for val, (i, j) in zip(pair_abs, pair_names):
        mat[i, j] = val
        mat[j, i] = val

    mask = np.tril(np.ones_like(mat, dtype=bool))
    masked = np.ma.array(mat, mask=mask)

    fig_size = max(8.2, n * 0.58)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    labels = [textwrap.fill(_paper_label(name).replace(": ", "\n"), width=10) for name in concept_names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    boundaries = []
    prev_family = _concept_family(concept_names[0]) if concept_names else None
    for idx, name in enumerate(concept_names[1:], start=1):
        fam = _concept_family(name)
        if fam != prev_family:
            boundaries.append(idx - 0.5)
        prev_family = fam
    for b in boundaries:
        ax.axhline(b, color="white", lw=1.3, alpha=0.95)
        ax.axvline(b, color="white", lw=1.3, alpha=0.95)

    if pair_abs.size > 0:
        top_idx = np.argsort(-pair_abs)[: min(18, len(pair_abs))]
        top_pairs = {pair_names[idx]: pair_abs[idx] for idx in top_idx}
    else:
        top_pairs = {}

    vmax = mat.max() if mat.size > 0 else 0.0
    for (i, j), v in top_pairs.items():
        color = "white" if vmax > 0 and v > vmax * 0.55 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute interaction", rotation=90)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] ga2m pair plot -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser("Analyze concept contributions for TrafficCBM")
    ap.add_argument("--ckpt_dir", default=None, help="directory containing best.pt and config.json")
    ap.add_argument("--ckpt", default=None, help="checkpoint path")
    ap.add_argument("--config", default=None, help="config.json path")
    ap.add_argument("--data_dir", required=True, help="data directory for the target dataset")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target", default="true", choices=["true", "pred"], help="which class logit to attribute")
    ap.add_argument("--baseline", default="mean", choices=["mean", "zero"], help="baseline value for ablation")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt_path, cfg_path, output_dir = _resolve_paths(args)
    os.makedirs(output_dir, exist_ok=True)
    cfg = _load_cfg(cfg_path)

    use_stat = cfg.get("use_stat", True)
    use_seq = cfg.get("use_seq", True)
    use_bytes = cfg.get("use_bytes", True)
    fusion_mode = cfg.get("fusion_mode", "6+1+1")

    label2id_path = os.path.join(args.data_dir, "label2id.json")
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): str(k) for k, v in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    print(f"[data_dir] {args.data_dir}", flush=True)

    ckpt_dir_for_scalers = args.ckpt_dir or (
        os.path.dirname(os.path.abspath(args.ckpt)) if getattr(args, "ckpt", None) else None
    )
    stat_s, seq_s = fit_or_load_scalers(
        args.data_dir,
        os.path.join(args.data_dir, "train_ids.txt"),
        fit_stat=use_stat,
        fit_seq=use_seq,
        ckpt_dir=ckpt_dir_for_scalers,
    )

    use_bytes, use_bytes_packet, use_bytes_cross = _resolve_bytes_branch_flags(
        fusion_mode,
        bool(cfg.get("use_bytes", False)),
        bool(cfg.get("use_bytes_packet", False)),
        bool(cfg.get("use_bytes_cross", False)),
    )

    bytes_per_packet = bool(cfg.get("bytes_per_packet", False) or fusion_mode in ("6+4+5", "6+4+5+3", "6+4+3"))
    num_packets = int(cfg.get("num_packets", 5))
    num_groups = int(cfg.get("num_groups", 3))
    ds = TrafficDataset(
        ids_txt=os.path.join(args.data_dir, f"{args.split}_ids.txt"),
        data_dir=args.data_dir,
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        max_seq_len=int(cfg.get("max_seq_len", 100)),
        bytes_len=int(cfg.get("bytes_len", 320)),
        bytes_per_packet=bytes_per_packet,
        bytes_pkt_len=int(cfg.get("bytes_pkt_len", 63)),
        num_packets=num_packets,
        stat_scaler=stat_s,
        seq_scaler=seq_s,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
    )
    print(f"[data] {args.split}: {len(ds)} samples", flush=True)

    num_classes = _num_classes(args.data_dir)
    stat_dim, _, _ = _get_stat_info(args.data_dir) if use_stat else (None, None, None)
    concept_mode = cfg.get("concept_mode", "grouped")
    concept_groups, concept_groups_src = _resolve_concept_groups(
        args.data_dir, args.ckpt_dir, use_stat, concept_mode
    )
    if use_stat and concept_mode == "grouped":
        if concept_groups_src == "saved":
            print("[concept_groups] loaded from checkpoint", flush=True)
        else:
            print(
                "[warn] concept_groups.json not found; rebuilding grouped stat feature order from current code/data. "
                "Old checkpoints may not be exactly reproducible.",
                flush=True,
            )
    model = build_model(
        num_classes=num_classes,
        stat_dim=stat_dim,
        num_concepts=cfg.get("num_concepts", 6),
        concept_mode=concept_mode,
        concept_groups=concept_groups,
        max_seq_len=cfg.get("max_seq_len", 100),
        bytes_len=cfg.get("bytes_len", 320),
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        use_bytes_packet=use_bytes_packet,
        use_bytes_cross=use_bytes_cross,
        head=cfg.get("head", "mlp"),
        stat_hidden=tuple(cfg.get("stat_hidden", [64, 64, 32])),
        seq_model_dim=cfg.get("seq_model_dim", 64),
        seq_num_heads=cfg.get("seq_num_heads", 4),
        seq_layers=cfg.get("seq_layers", 2),
        bytes_model_dim=cfg.get("bytes_model_dim", 64),
        bytes_num_heads=cfg.get("bytes_num_heads", 4),
        bytes_layers=cfg.get("bytes_layers", 2),
        bytes_arch=cfg.get("bytes_arch", "transformer"),
        seq_pooling=cfg.get("seq_pooling", "mean"),
        bytes_pooling=cfg.get("bytes_pooling", "mean"),
        fusion_mode=fusion_mode,
        dropout=cfg.get("dropout", 0.1),
        bytes_per_packet=bytes_per_packet,
        bytes_pkt_len=cfg.get("bytes_pkt_len", 63),
        num_packets=num_packets,
        num_groups=num_groups,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] loaded from {ckpt_path}", flush=True)
    print(f"[model] fusion_mode={fusion_mode} head={cfg.get('head', 'mlp')}", flush=True)

    logits, labels, concepts = collect_outputs(model, loader, device)
    pred = logits.argmax(dim=1)
    target_idx = labels if args.target == "true" else pred

    baseline = concepts.mean(dim=0) if args.baseline == "mean" else torch.zeros(concepts.size(1))
    concepts = concepts.to(device)
    target_idx = target_idx.to(device)
    baseline = baseline.to(device)

    deltas = ablation_contrib(model, concepts, target_idx, baseline).cpu()
    concept_names = _concept_names(
        fusion_mode,
        num_packets=num_packets,
        num_groups=num_groups,
        use_stat=use_stat,
        use_seq=use_seq,
        use_bytes=use_bytes,
        use_bytes_packet=use_bytes_packet,
        use_bytes_cross=use_bytes_cross,
    )

    prefix = os.path.join(output_dir, f"contrib_{args.split}")
    _write_ablation_report(
        prefix + "_ablation.txt",
        concept_names,
        deltas,
        labels.cpu(),
        id2label,
    )
    plot_ablation_bar(prefix + "_ablation_bar.png", concept_names, deltas)

    if isinstance(model.head, GA2MHead):
        exact = ga2m_exact_contrib(model, concepts, target_idx)
        if exact is not None:
            main_vals, pair_vals, pair_names = exact
            _write_ga2m_report(
                prefix + "_ga2m.txt",
                concept_names,
                main_vals,
                pair_vals,
                pair_names,
            )
            plot_ga2m_main_bar(prefix + "_ga2m_main_bar.png", concept_names, main_vals)
            plot_ga2m_shape_panel(
                prefix + "_ga2m_shape_panel.png",
                concept_names,
                concepts,
                main_vals,
                top_k=None,
            )
            plot_ga2m_shape_panel_selected(
                prefix + "_ga2m_shape_panel_selected.png",
                concept_names,
                concepts,
                main_vals,
            )
            plot_ga2m_shape_individuals(
                prefix + "_ga2m_shapes",
                concept_names,
                concepts,
                main_vals,
            )
            plot_ga2m_term_ranking(
                prefix + "_ga2m_term_ranking.png",
                concept_names,
                main_vals,
                pair_vals,
                pair_names,
            )
            plot_ga2m_pair_heatmap(
                prefix + "_ga2m_pair_heatmap.png",
                concept_names,
                pair_vals,
                pair_names,
            )

    np.save(prefix + "_concepts.npy", concepts.cpu().numpy())
    np.save(prefix + "_logits.npy", logits.numpy())
    np.save(prefix + "_labels.npy", labels.numpy())
    np.save(prefix + "_ablation_delta.npy", deltas.numpy())
    print(f"[save] raw arrays -> {prefix}_*.npy", flush=True)


if __name__ == "__main__":
    main()
