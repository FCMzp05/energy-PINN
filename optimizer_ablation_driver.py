#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    target_script: str
    output_root: str
    python_bin: str = "python"
    modes: str = (
        "adam_only,--optimizer adam;"
        "lbfgs_only,--optimizer lbfgs;"
        "adam_to_lbfgs,--optimizer hybrid;"
        "adam_to_lbfgs_ls,--optimizer hybrid_ls"
    )
    objectives: str = "dem,dcm"
    extra_args: str = ""
    run: int = 0


def parse_modes(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        if "," in item:
            label, args = item.split(",", 1)
            out.append((label.strip(), [arg for arg in args.strip().split() if arg]))
        else:
            out.append((item, []))
    return out


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Optimizer ablation driver for EJMA Fig. 3.")
    parser.add_argument("--target-script", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--modes", type=str, default=(
        "adam_only,--optimizer adam;"
        "lbfgs_only,--optimizer lbfgs;"
        "adam_to_lbfgs,--optimizer hybrid;"
        "adam_to_lbfgs_ls,--optimizer hybrid_ls"
    ))
    parser.add_argument("--objectives", type=str, default="dem,dcm")
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--run", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    output_root = args.output_root or str(Path(args.target_script).resolve().parent / "outputs" / "optimizer_ablation")
    return Config(
        target_script=args.target_script,
        output_root=output_root,
        python_bin=args.python_bin,
        modes=args.modes,
        objectives=args.objectives,
        extra_args=args.extra_args,
        run=args.run,
    )


def build_jobs(cfg: Config) -> list[dict[str, str]]:
    modes = parse_modes(cfg.modes)
    objectives = [item.strip() for item in cfg.objectives.split(",") if item.strip()]
    jobs: list[dict[str, str]] = []
    for obj in objectives:
        for label, extra_mode_args in modes:
            out_dir = Path(cfg.output_root) / f"{label}_{obj}"
            cmd = [cfg.python_bin, cfg.target_script, "--objective", obj, "--output-dir", str(out_dir)]
            if cfg.extra_args.strip():
                cmd.extend(cfg.extra_args.strip().split())
            cmd.extend(extra_mode_args)
            jobs.append(
                {
                    "label": label,
                    "objective": obj,
                    "output_dir": str(out_dir),
                    "command": " ".join(cmd),
                }
            )
    return jobs


def main() -> None:
    cfg = parse_args()
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(cfg)

    with (out_root / "optimizer_ablation_plan.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "jobs": jobs}, f, indent=2)
    with (out_root / "optimizer_ablation_plan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "objective", "output_dir", "command"])
        writer.writeheader()
        writer.writerows(jobs)

    if cfg.run == 0:
        print("Saved outputs:")
        print("  optimizer_ablation_plan.json")
        print("  optimizer_ablation_plan.csv")
        return

    summary_rows: list[dict[str, str | int | float]] = []
    for idx, job in enumerate(jobs, start=1):
        print(f"[Run] {idx:02d}/{len(jobs)} | {job['label']} | {job['objective']}")
        tick = time.time()
        proc = subprocess.run(job["command"], shell=True, check=False)
        summary_rows.append(
            {
                "index": idx,
                "label": job["label"],
                "objective": job["objective"],
                "returncode": proc.returncode,
                "elapsed_sec": time.time() - tick,
                "output_dir": job["output_dir"],
            }
        )

    with (out_root / "optimizer_ablation_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "objective", "returncode", "elapsed_sec", "output_dir"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Saved outputs:")
    print("  optimizer_ablation_plan.json")
    print("  optimizer_ablation_plan.csv")
    print("  optimizer_ablation_summary.csv")


if __name__ == "__main__":
    main()
