#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    workdir: Path
    command: list[str]
    log_name: str


def py_task(name: str, workdir: Path, python_bin: str, script: str, args: list[str]) -> Task:
    return Task(
        name=name,
        workdir=workdir,
        command=[python_bin, str(workdir / script), *args],
        log_name=f"{name}.log",
    )


def jl_task(name: str, workdir: Path, julia_bin: str, script: str, args: list[str]) -> Task:
    return Task(
        name=name,
        workdir=workdir,
        command=[julia_bin, str(workdir / script), *args],
        log_name=f"{name}.log",
    )


def build_elastic_tasks(root: Path, python_bin: str, julia_bin: str, output_root: Path) -> list[Task]:
    elastic_root = root / "elastic"

    def ep(name: str, subdir: str, script: str, args: list[str], rel_out: str) -> Task:
        workdir = elastic_root / subdir
        return py_task(name, workdir, python_bin, script, [*args, "--output-dir", str(output_root / "elastic" / rel_out)])

    def ej(name: str, subdir: str, script: str, args: list[str], rel_out: str) -> Task:
        workdir = elastic_root / subdir
        return jl_task(name, workdir, julia_bin, script, [*args, "--output-dir", str(output_root / "elastic" / rel_out)])

    return [
        ej("01_elastic_cantilever_concentrated_force_fem", "cantilever_beam", "cantilever_beam_concentrated_force_fem.jl", [], "cantilever_beam/concentrated_force/fem"),
        ep("02_elastic_cantilever_concentrated_force_feinn_dem", "cantilever_beam", "cantilever_beam_concentrated_force_feinn_baseline.py", ["--objective", "dem"], "cantilever_beam/concentrated_force/feinn_dem"),
        ep("03_elastic_cantilever_concentrated_force_feinn_dcm", "cantilever_beam", "cantilever_beam_concentrated_force_feinn_baseline.py", ["--objective", "dcm"], "cantilever_beam/concentrated_force/feinn_dcm"),
        ej("04_elastic_cantilever_distributed_displacement_fem", "cantilever_beam", "cantilever_beam_distributed_displacement_fem.jl", [], "cantilever_beam/distributed_displacement/fem"),
        ep("05_elastic_cantilever_distributed_displacement_feinn_dem", "cantilever_beam", "cantilever_beam_distributed_displacement_feinn_baseline.py", ["--objective", "dem"], "cantilever_beam/distributed_displacement/feinn_dem"),
        ep("06_elastic_cantilever_distributed_displacement_feinn_dcm", "cantilever_beam", "cantilever_beam_distributed_displacement_feinn_baseline.py", ["--objective", "dcm"], "cantilever_beam/distributed_displacement/feinn_dcm"),
        ej("07_elastic_defected_plate_fem", "defected_plate", "defected_plate_fem.jl", [], "defected_plate/fem"),
        ep("08_elastic_defected_plate_feinn_dem", "defected_plate", "defected_plate_feinn_baseline.py", ["--objective", "dem"], "defected_plate/feinn_dem"),
        ep("09_elastic_defected_plate_feinn_dcm", "defected_plate", "defected_plate_feinn_baseline.py", ["--objective", "dcm"], "defected_plate/feinn_dcm"),
        ej("10_elastic_footing_displacement_fem", "footing_cases", "footing_cases_fem.jl", ["--case-name", "displacement"], "footing_cases/fem/displacement"),
        ej("11_elastic_footing_force_fem", "footing_cases", "footing_cases_fem.jl", ["--case-name", "force"], "footing_cases/fem/force"),
        ep("12_elastic_footing_feinn_dem", "footing_cases", "footing_cases_feinn_baseline.py", ["--case", "both", "--objective", "dem"], "footing_cases/feinn_dem"),
        ep("13_elastic_footing_feinn_dcm", "footing_cases", "footing_cases_feinn_baseline.py", ["--case", "both", "--objective", "dcm"], "footing_cases/feinn_dcm"),
        ej("14_elastic_heterogeneous_fem", "heterogeneous_elastic", "heterogeneous_elastic_fem.jl", [], "heterogeneous_elastic/fem"),
        ep("15_elastic_heterogeneous_feinn_dem", "heterogeneous_elastic", "heterogeneous_elastic_feinn_baseline.py", ["--objective", "dem"], "heterogeneous_elastic/feinn_dem"),
        ep("16_elastic_heterogeneous_feinn_dcm", "heterogeneous_elastic", "heterogeneous_elastic_feinn_baseline.py", ["--objective", "dcm"], "heterogeneous_elastic/feinn_dcm"),
        ej("17_elastic_multimaterial_fem", "multimaterial_elastic", "multimaterial_elastic_fem.jl", [], "multimaterial_elastic/fem"),
        ep("18_elastic_multimaterial_feinn_dem", "multimaterial_elastic", "multimaterial_elastic_feinn_baseline.py", ["--objective", "dem"], "multimaterial_elastic/feinn_dem"),
        ep("19_elastic_multimaterial_feinn_dcm", "multimaterial_elastic", "multimaterial_elastic_feinn_baseline.py", ["--objective", "dcm"], "multimaterial_elastic/feinn_dcm"),
    ]


def build_plastic_tasks(root: Path, python_bin: str, julia_bin: str, output_root: Path) -> list[Task]:
    cantilever_root = root / "plastic" / "cantilever_beam"
    plate_root = root / "plastic" / "perforated_plate"
    return [
        jl_task("20_plastic_cantilever_force_fem", cantilever_root, julia_bin, "cantilever_beam_force_fem_julia.jl", ["--output-dir", str(output_root / "plastic" / "cantilever_beam" / "force" / "fem")]),
        py_task("21_plastic_cantilever_force_feinn_dem", cantilever_root, python_bin, "cantilever_beam_force_feinn_baseline.py", ["--objective", "dem", "--fem-dir", str(output_root / "plastic" / "cantilever_beam" / "force" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "force" / "feinn_dem")]),
        py_task("22_plastic_cantilever_force_feinn_dcm", cantilever_root, python_bin, "cantilever_beam_force_feinn_baseline.py", ["--objective", "dcm", "--fem-dir", str(output_root / "plastic" / "cantilever_beam" / "force" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "force" / "feinn_dcm")]),
        jl_task("23_plastic_cantilever_displacement_fem", cantilever_root, julia_bin, "cantilever_beam_displacement_fem_julia.jl", ["--output-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "fem")]),
        py_task("24_plastic_cantilever_displacement_feinn_dem", cantilever_root, python_bin, "cantilever_beam_displacement_feinn_baseline.py", ["--objective", "dem", "--fem-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "feinn_dem")]),
        py_task("25_plastic_cantilever_displacement_feinn_dcm", cantilever_root, python_bin, "cantilever_beam_displacement_feinn_baseline.py", ["--objective", "dcm", "--fem-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "feinn_dcm")]),
        jl_task("26_plastic_perforated_square_fem", plate_root, julia_bin, "perforated_plate_square_fem_julia.jl", ["--hardening-mode", "isotropic", "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem")]),
        py_task("27_plastic_perforated_square_feinn_isotropic_dem", plate_root, python_bin, "perforated_plate_square_feinn_baseline.py", ["--objective", "dem", "--hardening", "isotropic", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "feinn_isotropic_dem")]),
        py_task("28_plastic_perforated_square_feinn_isotropic_dcm", plate_root, python_bin, "perforated_plate_square_feinn_baseline.py", ["--objective", "dcm", "--hardening", "isotropic", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "feinn_isotropic_dcm")]),
        jl_task("29_plastic_perforated_square_fem_kinematic", plate_root, julia_bin, "perforated_plate_square_fem_julia.jl", ["--hardening-mode", "kinematic", "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem_kinematic")]),
        py_task("30_plastic_perforated_square_feinn_kinematic_dem", plate_root, python_bin, "perforated_plate_square_feinn_baseline.py", ["--objective", "dem", "--hardening", "kinematic", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem_kinematic"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "feinn_kinematic_dem")]),
        py_task("31_plastic_perforated_square_feinn_kinematic_dcm", plate_root, python_bin, "perforated_plate_square_feinn_baseline.py", ["--objective", "dcm", "--hardening", "kinematic", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem_kinematic"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "feinn_kinematic_dcm")]),
        py_task("32_plastic_perforated_square_inference", plate_root, python_bin, "perforated_plate_square_feinn_inference.py", ["--source-fields", str(output_root / "plastic" / "perforated_plate" / "square" / "feinn_isotropic_dem" / "perforated_plate_square_feinn_baseline_fields.csv"), "--target-fem-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "fem"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "square" / "inference")]),
        jl_task("33_plastic_perforated_rectangular_fem", plate_root, julia_bin, "perforated_plate_rectangular_fem_julia.jl", ["--output-dir", str(output_root / "plastic" / "perforated_plate" / "rectangular" / "fem")]),
        py_task("34_plastic_perforated_rectangular_feinn_dem", plate_root, python_bin, "perforated_plate_rectangular_feinn_baseline.py", ["--objective", "dem", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "rectangular" / "fem"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "rectangular" / "feinn_dem")]),
        py_task("35_plastic_perforated_rectangular_feinn_dcm", plate_root, python_bin, "perforated_plate_rectangular_feinn_baseline.py", ["--objective", "dcm", "--fem-dir", str(output_root / "plastic" / "perforated_plate" / "rectangular" / "fem"), "--output-dir", str(output_root / "plastic" / "perforated_plate" / "rectangular" / "feinn_dcm")]),
    ]


def build_extension_tasks(root: Path, python_bin: str, julia_bin: str, output_root: Path) -> list[Task]:
    method_root = root / "method_study"
    bimaterial_root = root / "bimaterial"
    mesh_root = root / "mesh"
    notched_root = root / "notched"
    cantilever_root = root / "plastic" / "cantilever_beam"
    plate_root = root / "plastic" / "perforated_plate"
    return [
        py_task("36_method_optimizer_ablation", method_root, python_bin, "optimizer_ablation_driver.py", ["--target-script", str(plate_root / "perforated_plate_square_feinn_baseline.py"), "--output-root", str(output_root / "method_study" / "optimizer_ablation")]),
        py_task("37_method_step_inheritance", method_root, python_bin, "step_inheritance_driver.py", ["--target-script", str(cantilever_root / "cantilever_beam_displacement_feinn_baseline.py"), "--output-root", str(output_root / "method_study" / "step_inheritance")]),
        jl_task("38_bimaterial_fem", bimaterial_root, julia_bin, "bimaterial_shear_fem.jl", ["--output-dir", str(output_root / "bimaterial" / "fem")]),
        py_task("39_bimaterial_feinn_dem", bimaterial_root, python_bin, "bimaterial_shear_feinn.py", ["--objective", "dem", "--output-dir", str(output_root / "bimaterial" / "feinn_dem")]),
        py_task("40_bimaterial_feinn_dcm", bimaterial_root, python_bin, "bimaterial_shear_feinn.py", ["--objective", "dcm", "--output-dir", str(output_root / "bimaterial" / "feinn_dcm")]),
        jl_task("41_notched_v_notch_fem", notched_root, julia_bin, "v_notch_fem_julia.jl", ["--output-dir", str(output_root / "notched" / "v_notch" / "fem")]),
        py_task("42_notched_v_notch_feinn_dem", notched_root, python_bin, "v_notch_feinn.py", ["--objective", "dem", "--fem-dir", str(output_root / "notched" / "v_notch" / "fem"), "--output-dir", str(output_root / "notched" / "v_notch" / "feinn_dem")]),
        py_task("43_notched_v_notch_feinn_dcm", notched_root, python_bin, "v_notch_feinn.py", ["--objective", "dcm", "--fem-dir", str(output_root / "notched" / "v_notch" / "fem"), "--output-dir", str(output_root / "notched" / "v_notch" / "feinn_dcm")]),
        jl_task("44_notched_v_notch_cyclic_fem", notched_root, julia_bin, "v_notch_cyclic_fem_julia.jl", ["--output-dir", str(output_root / "notched" / "v_notch_cyclic" / "fem")]),
        py_task("45_notched_v_notch_cyclic_feinn_dem", notched_root, python_bin, "v_notch_cyclic_feinn.py", ["--objective", "dem", "--mesh-dir", str(output_root / "notched" / "v_notch" / "fem"), "--history-dir", str(output_root / "notched" / "v_notch_cyclic" / "fem"), "--output-dir", str(output_root / "notched" / "v_notch_cyclic" / "feinn_dem")]),
        py_task("46_notched_v_notch_cyclic_feinn_dcm", notched_root, python_bin, "v_notch_cyclic_feinn.py", ["--objective", "dcm", "--mesh-dir", str(output_root / "notched" / "v_notch" / "fem"), "--history-dir", str(output_root / "notched" / "v_notch_cyclic" / "fem"), "--output-dir", str(output_root / "notched" / "v_notch_cyclic" / "feinn_dcm")]),
        jl_task("47_cantilever_3d_bridge_fem", cantilever_root, julia_bin, "cantilever_beam_3d_fem_julia.jl", ["--source-2d-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "bridge_3d_fem")]),
        py_task("48_cantilever_3d_bridge_feinn", cantilever_root, python_bin, "cantilever_beam_3d_feinn.py", ["--source-2d-fields", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "feinn_dem" / "cantilever_beam_displacement_feinn_baseline_fields.csv"), "--source-2d-fem-dir", str(output_root / "plastic" / "cantilever_beam" / "displacement" / "fem"), "--output-dir", str(output_root / "plastic" / "cantilever_beam" / "bridge_3d_feinn")]),
        jl_task("49_mesh_square_isotropic_fem", mesh_root, julia_bin, "run_square_isotropic_mesh_sensitivity_fem.jl", ["--paper-root", str(root), "--output-root", str(output_root / "mesh" / "square_isotropic_mesh_sensitivity")]),
        py_task("50_mesh_square_isotropic_feinn_dem", mesh_root, python_bin, "run_square_isotropic_mesh_sensitivity_feinn.py", ["--paper-root", str(root), "--fem-root", str(output_root / "mesh" / "square_isotropic_mesh_sensitivity" / "fem"), "--output-root", str(output_root / "mesh" / "square_isotropic_mesh_sensitivity"), "--python-bin", python_bin, "--objective", "dem"]),
    ]


def build_tasks(root: Path, python_bin: str, julia_bin: str, output_root: Path, core_only: bool) -> list[Task]:
    tasks: list[Task] = []
    tasks.extend(build_elastic_tasks(root, python_bin, julia_bin, output_root))
    tasks.extend(build_plastic_tasks(root, python_bin, julia_bin, output_root))
    if not core_only:
        tasks.extend(build_extension_tasks(root, python_bin, julia_bin, output_root))
    return tasks


def stream_process(task: Task, log_path: Path) -> tuple[int, float]:
    start = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"WORKDIR: {task.workdir}\n")
        log_file.write(f"COMMAND: {' '.join(task.command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            task.command,
            cwd=task.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        returncode = process.wait()
    return returncode, time.time() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all available paper_code tasks sequentially from one top-level script.")
    parser.add_argument("--output-root", type=str, default=None, help="Default: paper_code/outputs/run_all_paper_code")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--julia-bin", type=str, default="julia")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_root = Path(args.output_root) if args.output_root else root / "outputs" / "run_all_paper_code"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(root, args.python_bin, args.julia_bin, output_root, args.core_only)
    if args.start_index > 1:
        tasks = tasks[args.start_index - 1 :]

    print(f"Batch output root: {output_root}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Core only: {args.core_only}")
    print(f"Start index: {args.start_index}")

    if args.dry_run:
        for task in tasks:
            print(f"{task.name}: {' '.join(task.command)}")
        return

    summary_rows: list[dict[str, str]] = []
    batch_start = time.time()
    for idx, task in enumerate(tasks, start=1):
        print("=" * 80)
        print(f"[{idx:02d}/{len(tasks):02d}] START {task.name}")
        print(f"WORKDIR: {task.workdir}")
        print(f"COMMAND: {' '.join(task.command)}")
        log_path = logs_dir / task.log_name
        returncode, elapsed = stream_process(task, log_path)
        status = "ok" if returncode == 0 else "failed"
        print(f"[{idx:02d}/{len(tasks):02d}] END {task.name} | status={status} | elapsed_sec={elapsed:.2f}")
        summary_rows.append(
            {
                "task": task.name,
                "status": status,
                "returncode": str(returncode),
                "elapsed_sec": f"{elapsed:.2f}",
                "log_path": str(log_path),
                "command": " ".join(task.command),
            }
        )
        if returncode != 0 and not args.continue_on_error:
            break

    summary_path = output_root / "run_all_paper_code_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "status", "returncode", "elapsed_sec", "log_path", "command"])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_elapsed = time.time() - batch_start
    print("=" * 80)
    print(f"Batch finished in {total_elapsed:.2f} seconds")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
