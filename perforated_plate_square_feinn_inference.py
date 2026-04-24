#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


def resolve_source_fields(path: Path) -> Path:
    if path.exists():
        return path
    candidates = [
        path.with_name("perforated_plate_square_feinn_baseline_fields.csv"),
        path.with_name("perforated_plate_square_feinn_fields.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    sibling_matches = sorted(path.parent.glob("*fields.csv"))
    for candidate in sibling_matches:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Source fields file not found: {path}")


def idw_transfer(
    src_xy: np.ndarray,
    src_val: np.ndarray,
    dst_xy: np.ndarray,
    k_nearest: int,
    power: float,
) -> np.ndarray:
    out = np.zeros(dst_xy.shape[0], dtype=np.float64)
    for i, pt in enumerate(dst_xy):
        dist = np.sqrt(np.sum((src_xy - pt[None, :]) ** 2, axis=1))
        order = np.argsort(dist)[:k_nearest]
        dist_sel = np.maximum(dist[order], 1.0e-12)
        weight = 1.0 / (dist_sel**power)
        out[i] = np.sum(weight * src_val[order]) / np.sum(weight)
    return out


def l2_rel(ref: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(ref**2)))
    if denom < 1.0e-12:
        return 0.0
    return float(np.sqrt(np.sum((pred - ref) ** 2)) / denom)


def save_compare_plot(
    nodes: np.ndarray,
    triangles: np.ndarray,
    ref_fields: dict[str, np.ndarray],
    pred_fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    names = [("uy", "UY"), ("sigma_vm", "Von Mises"), ("eps_p_eq", "Eq. plastic strain")]
    fig, axes = plt.subplots(len(names), 3, figsize=(10.0, 12.5), constrained_layout=True)
    for i, (key, title) in enumerate(names):
        vmin = min(float(np.min(ref_fields[key])), float(np.min(pred_fields[key])))
        vmax = max(float(np.max(ref_fields[key])), float(np.max(pred_fields[key])))
        err = pred_fields[key] - ref_fields[key]
        m0 = axes[i, 0].tripcolor(tri, ref_fields[key], shading="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Target FEM: {title}")
        fig.colorbar(m0, ax=axes[i, 0], shrink=0.8)
        m1 = axes[i, 1].tripcolor(tri, pred_fields[key], shading="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Inferred: {title}")
        fig.colorbar(m1, ax=axes[i, 1], shrink=0.8)
        m2 = axes[i, 2].tripcolor(tri, err, shading="gouraud", cmap="coolwarm")
        axes[i, 2].set_title(f"Error: {title}")
        fig.colorbar(m2, ax=axes[i, 2], shrink=0.8)
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-mesh inference for square perforated-plate FEINN fields.")
    parser.add_argument("--source-fields", type=str, required=True)
    parser.add_argument("--target-fem-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--k-nearest", type=int, default=6)
    parser.add_argument("--idw-power", type=float, default=2.0)
    args = parser.parse_args()

    source_path = resolve_source_fields(Path(args.source_fields))
    target_dir = Path(args.target_fem_dir)
    out_dir = Path(args.output_dir) if args.output_dir else target_dir.parent / "perforated_plate_square_feinn_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_df = pd.read_csv(source_path)
    ref_df = pd.read_csv(target_dir / "perforated_plate_square_fem_fields.csv")
    elem_df = pd.read_csv(target_dir / "perforated_plate_square_fem_elements.csv")
    nodes = ref_df[["x", "y"]].to_numpy(dtype=np.float64)
    triangles = elem_df[["n1", "n2", "n3"]].to_numpy(dtype=np.int64) - 1
    src_xy = src_df[["x", "y"]].to_numpy(dtype=np.float64)
    dst_xy = ref_df[["x", "y"]].to_numpy(dtype=np.float64)

    field_pairs = {
        "ux": ("ux_feinn", "ux"),
        "uy": ("uy_feinn", "uy"),
        "sigma_xx": ("sigma_xx_feinn", "sigma_xx"),
        "sigma_yy": ("sigma_yy_feinn", "sigma_yy"),
        "sigma_xy": ("sigma_xy_feinn", "sigma_xy"),
        "sigma_vm": ("sigma_vm_feinn", "sigma_vm"),
        "eps_p_eq": ("eps_p_eq_feinn", "eps_p_eq"),
    }

    pred_fields: dict[str, np.ndarray] = {}
    ref_fields: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for key, (src_col, ref_col) in field_pairs.items():
        pred = idw_transfer(src_xy, src_df[src_col].to_numpy(dtype=np.float64), dst_xy, args.k_nearest, args.idw_power)
        ref = ref_df[ref_col].to_numpy(dtype=np.float64)
        pred_fields[key] = pred
        ref_fields[key] = ref
        metrics[key] = {
            "mae": float(np.mean(np.abs(pred - ref))),
            "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "l2_rel": l2_rel(ref, pred),
        }

    pd.DataFrame(
        {
            "x": nodes[:, 0],
            "y": nodes[:, 1],
            **{f"{key}_target": value for key, value in ref_fields.items()},
            **{f"{key}_inferred": value for key, value in pred_fields.items()},
        }
    ).to_csv(out_dir / "perforated_plate_square_feinn_inference_fields.csv", index=False)
    with (out_dir / "perforated_plate_square_feinn_inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_compare_plot(nodes, triangles, ref_fields, pred_fields, out_dir / "perforated_plate_square_feinn_inference_panel.png")

    print("Saved outputs:")
    print("  perforated_plate_square_feinn_inference_fields.csv")
    print("  perforated_plate_square_feinn_inference_metrics.json")
    print("  perforated_plate_square_feinn_inference_panel.png")


if __name__ == "__main__":
    main()
