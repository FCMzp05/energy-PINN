#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    paper_root: str = ""
    fem_root: str = ""
    output_root: str = ""
    python_bin: str = sys.executable
    objective: str = "dem"
    rprop_epochs: int | None = None
    lr: float | None = None
    width_nn: int | None = None
    blocks: int | None = None
    dem_residual_weight: float | None = None
    device: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Square isotropic mesh-sensitivity FEINN runner.")
    parser.add_argument("--paper-root", type=str, default=None)
    parser.add_argument("--fem-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--objective", type=str, choices=("dem", "dcm"), default="dem")
    parser.add_argument("--rprop-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--width-nn", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--dem-residual-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    paper_root_default = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root) if args.output_root else paper_root_default / "outputs" / "mesh" / "square_isotropic_mesh_sensitivity"
    fem_root = Path(args.fem_root) if args.fem_root else output_root / "fem"
    paper_root = Path(args.paper_root) if args.paper_root else paper_root_default
    return Config(
        paper_root=str(paper_root),
        fem_root=str(fem_root),
        output_root=str(output_root),
        python_bin=args.python_bin,
        objective=args.objective,
        rprop_epochs=args.rprop_epochs,
        lr=args.lr,
        width_nn=args.width_nn,
        blocks=args.blocks,
        dem_residual_weight=args.dem_residual_weight,
        device=args.device,
    )


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_last_rel_residual(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["rel_residual"]) if rows else float("nan")


def main() -> None:
    cfg = build_config(parse_args())
    feinn_script = Path(cfg.paper_root) / "plastic" / "perforated_plate" / "perforated_plate_square_feinn_baseline.py"
    if not feinn_script.exists():
        raise FileNotFoundError(f"Missing FEINN script: {feinn_script}")

    plan_path = Path(cfg.output_root) / "mesh_sensitivity_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing FEM mesh plan: {plan_path}")

    cases = load_plan(plan_path)
    out_root = Path(cfg.output_root) / f"feinn_{cfg.objective}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | float | int]] = []
    for case in cases:
        label = case["label"]
        fem_dir = Path(case["fem_dir"])
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            cfg.python_bin,
            str(feinn_script),
            "--objective",
            cfg.objective,
            "--hardening",
            "isotropic",
            "--fem-dir",
            str(fem_dir),
            "--output-dir",
            str(out_dir),
        ]
        if cfg.rprop_epochs is not None:
            cmd += ["--rprop-epochs", str(cfg.rprop_epochs)]
        if cfg.lr is not None:
            cmd += ["--lr", str(cfg.lr)]
        if cfg.width_nn is not None:
            cmd += ["--width-nn", str(cfg.width_nn)]
        if cfg.blocks is not None:
            cmd += ["--blocks", str(cfg.blocks)]
        if cfg.dem_residual_weight is not None:
            cmd += ["--dem-residual-weight", str(cfg.dem_residual_weight)]
        if cfg.device is not None:
            cmd += ["--device", cfg.device]

        print(f"Running FEINN mesh case {label} | objective={cfg.objective}")
        tick = time.time()
        subprocess.run(cmd, check=True)
        elapsed_sec = time.time() - tick

        metrics_path = out_dir / "perforated_plate_square_feinn_baseline_metrics.json"
        history_path = out_dir / "perforated_plate_square_feinn_baseline_training_history.csv"
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        summary_rows.append(
            {
                "label": label,
                "ntheta": int(case["ntheta"]),
                "nradial": int(case["nradial"]),
                "nnodes": int(case["nnodes"]),
                "nelements": int(case["nelements"]),
                "objective": cfg.objective,
                "elapsed_sec": elapsed_sec,
                "uy_rmse": float(metrics["uy_rmse"]),
                "sigma_yy_rmse": float(metrics["sigma_yy_rmse"]),
                "sigma_vm_rmse": float(metrics["sigma_vm_rmse"]),
                "eps_p_eq_rmse": float(metrics["eps_p_eq_rmse"]),
                "final_rel_residual": load_last_rel_residual(history_path),
                "fem_dir": str(fem_dir),
                "feinn_dir": str(out_dir),
            }
        )

    summary_path = out_root / "mesh_sensitivity_feinn_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "ntheta",
                "nradial",
                "nnodes",
                "nelements",
                "objective",
                "elapsed_sec",
                "uy_rmse",
                "sigma_yy_rmse",
                "sigma_vm_rmse",
                "eps_p_eq_rmse",
                "final_rel_residual",
                "fem_dir",
                "feinn_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
