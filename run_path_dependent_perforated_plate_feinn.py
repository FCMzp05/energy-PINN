#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch


@dataclass
class Config:
    fem_dir: str = ""
    objective_mode: str = "dcm"
    hardening_mode: str = "isotropic"
    path_case: str = "case1"
    width: float = 200.0
    height: float = 200.0
    radius: float = 50.0
    thickness: float = 100.0
    young: float = 7.0e4
    poisson: float = 0.20
    yield_stress: float = 250.0
    tangent_modulus: float = 2171.0
    iso_q1: float = -216.9135
    iso_b1: float = 213.9273
    kin_c1: float = 58791.656
    kin_gamma1: float = 147.7362
    kin_c2: float = 1803.7759
    kin_gamma2: float = 0.0
    load_steps: int = 21
    rprop_epochs: int = 4000
    lr: float = 5.0e-2
    width_nn: int = 96
    blocks: int = 6
    dem_residual_weight: float = 1.0
    reg_weight: float = 1.0e-10
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_scalar(text: str) -> float | int | str:
    value = text.strip()
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_run_config(path: Path) -> dict[str, float | int | str]:
    cfg: dict[str, float | int | str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = parse_scalar(value)
    return cfg


def load_node_ids(path: Path) -> np.ndarray:
    rows = pd.read_csv(path)
    return rows["node_id"].to_numpy(dtype=np.int64) - 1


def load_top_edges(path: Path) -> np.ndarray:
    rows = pd.read_csv(path)
    return rows[["n1", "n2"]].to_numpy(dtype=np.int64) - 1


def load_fem_dataset(fem_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    field_rows = pd.read_csv(fem_dir / "path_perforated_plate_fem_fields.csv")
    elem_rows = pd.read_csv(fem_dir / "path_perforated_plate_fem_elements.csv")
    order = np.argsort(field_rows["node_id"].to_numpy(dtype=np.int64))
    nodes = field_rows.loc[order, ["x", "y"]].to_numpy(dtype=np.float64)
    triangles = elem_rows[["n1", "n2", "n3"]].to_numpy(dtype=np.int64) - 1
    mesh = {
        "nodes": nodes,
        "triangles": triangles,
        "left_nodes": load_node_ids(fem_dir / "path_perforated_plate_fem_boundary_left.csv"),
        "bottom_nodes": load_node_ids(fem_dir / "path_perforated_plate_fem_boundary_bottom.csv"),
        "top_nodes": load_node_ids(fem_dir / "path_perforated_plate_fem_boundary_top.csv"),
        "right_nodes": load_node_ids(fem_dir / "path_perforated_plate_fem_boundary_right.csv"),
        "hole_nodes": load_node_ids(fem_dir / "path_perforated_plate_fem_boundary_hole.csv"),
        "top_edges": load_top_edges(fem_dir / "path_perforated_plate_fem_top_edges.csv"),
    }
    fem_fields = {
        "ux": field_rows.loc[order, "ux"].to_numpy(dtype=np.float64),
        "uy": field_rows.loc[order, "uy"].to_numpy(dtype=np.float64),
        "sigma_xx": field_rows.loc[order, "sigma_xx"].to_numpy(dtype=np.float64),
        "sigma_yy": field_rows.loc[order, "sigma_yy"].to_numpy(dtype=np.float64),
        "sigma_xy": field_rows.loc[order, "sigma_xy"].to_numpy(dtype=np.float64),
        "sigma_vm": field_rows.loc[order, "sigma_vm"].to_numpy(dtype=np.float64),
        "eps_p_eq": field_rows.loc[order, "eps_p_eq"].to_numpy(dtype=np.float64),
    }
    path_df = pd.read_csv(fem_dir / "path_perforated_plate_fem_path.csv")
    curve_df = pd.read_csv(fem_dir / "path_perforated_plate_fem_load_curve.csv")
    return mesh, fem_fields, path_df, curve_df


def validate_path_data(cfg: Config, path_df: pd.DataFrame, curve_df: pd.DataFrame) -> None:
    required_cols = {"load_step", "time", "top_uy"}
    if set(path_df.columns) != required_cols:
        raise ValueError(f"Unexpected path columns: {list(path_df.columns)}")
    curve_cols = {"load_step", "time", "top_uy", "reaction_y"}
    if set(curve_df.columns) != curve_cols:
        raise ValueError(f"Unexpected load-curve columns: {list(curve_df.columns)}")
    if len(path_df) != cfg.load_steps:
        raise ValueError(f"Path length {len(path_df)} does not match load_steps={cfg.load_steps}")
    if len(curve_df) != cfg.load_steps:
        raise ValueError(f"Load-curve length {len(curve_df)} does not match load_steps={cfg.load_steps}")
    if not np.array_equal(path_df["load_step"].to_numpy(dtype=np.int64), np.arange(1, cfg.load_steps + 1, dtype=np.int64)):
        raise ValueError("Path load_step sequence is invalid")
    if not np.array_equal(curve_df["load_step"].to_numpy(dtype=np.int64), np.arange(1, cfg.load_steps + 1, dtype=np.int64)):
        raise ValueError("Load-curve load_step sequence is invalid")
    if not np.allclose(path_df["top_uy"].to_numpy(dtype=np.float64), curve_df["top_uy"].to_numpy(dtype=np.float64)):
        raise ValueError("FEM path and FEM load curve use different top_uy histories")


def elastic_constants(cfg: Config) -> tuple[float, float]:
    lam = cfg.young * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    return lam, mu


def hardening_modulus_from_tangent(cfg: Config) -> float:
    ce = cfg.young
    ct = cfg.tangent_modulus
    if abs(ce - ct) < 1.0e-12:
        return 0.0
    return ct * ce / (ce - ct)


def elastic_matrix_np(cfg: Config) -> np.ndarray:
    lam, mu = elastic_constants(cfg)
    return np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, 2.0 * mu],
        ],
        dtype=np.float64,
    )


def build_tri_operators(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b_mats = []
    areas = []
    for tri in triangles:
        xy = nodes[tri]
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(area2)
        beta = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
        gamma = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
        dndx = beta / area2
        dndy = gamma / area2
        b = np.zeros((4, 6), dtype=np.float64)
        for a in range(3):
            b[0, 2 * a] = dndx[a]
            b[1, 2 * a + 1] = dndy[a]
            b[3, 2 * a] = 0.5 * dndy[a]
            b[3, 2 * a + 1] = 0.5 * dndx[a]
        b_mats.append(b)
        areas.append(area)
    return np.stack(b_mats), np.array(areas, dtype=np.float64)


def build_dof_map(triangles: np.ndarray) -> np.ndarray:
    out = np.zeros((triangles.shape[0], 6), dtype=np.int64)
    for e, tri in enumerate(triangles):
        dofs = []
        for nid in tri:
            dofs.extend([2 * int(nid), 2 * int(nid) + 1])
        out[e] = np.array(dofs, dtype=np.int64)
    return out


def nodal_average(nnodes: int, triangles: np.ndarray, elem_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros((nnodes, elem_values.shape[1]), dtype=np.float64)
    counts = np.zeros(nnodes, dtype=np.float64)
    for e, tri in enumerate(triangles):
        nodal[tri] += elem_values[e]
        counts[tri] += 1.0
    counts[counts == 0.0] = 1.0
    return nodal / counts[:, None]


def von_mises_from_stress(stress: np.ndarray) -> np.ndarray:
    sx = stress[:, 0]
    sy = stress[:, 1]
    sz = stress[:, 2]
    sxy = stress[:, 3]
    return np.sqrt(np.maximum(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * sxy**2, 0.0))


def objective_suffix(mode: str) -> str:
    if mode not in {"dem", "dcm"}:
        raise ValueError(f"Unknown objective mode: {mode}")
    return mode


def energy_scale(cfg: Config) -> float:
    return max(cfg.young * cfg.width * cfg.height * cfg.thickness, 1.0)


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def voigt_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2] + 2.0 * a[:, 3] * b[:, 3]


class ResBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.fc1(x))
        y = self.fc2(y)
        return self.act(x + y)


class ResNet(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, blocks: int) -> None:
        super().__init__()
        self.in_layer = torch.nn.Linear(in_dim, width)
        self.blocks = torch.nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.out_layer = torch.nn.Linear(width, out_dim)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.in_layer(x))
        for block in self.blocks:
            y = block(y)
        return self.out_layer(y)


def normalize_coords(cfg: Config, nodes_t: torch.Tensor) -> torch.Tensor:
    out = nodes_t.clone()
    out[:, 0] = 2.0 * nodes_t[:, 0] / cfg.width - 1.0
    out[:, 1] = 2.0 * nodes_t[:, 1] / cfg.height - 1.0
    return out


def apply_hard_bc(cfg: Config, nodes_t: torch.Tensor, raw_out: torch.Tensor, target_uy: float) -> torch.Tensor:
    xh = nodes_t[:, 0:1] / cfg.width
    yh = nodes_t[:, 1:2] / cfg.height
    bubble_y = yh * (1.0 - yh)
    ux = cfg.width * xh * raw_out[:, 0:1]
    uy = target_uy * yh + cfg.height * bubble_y * raw_out[:, 1:2]
    return torch.cat([ux, uy], dim=1)


def build_torch_data(cfg: Config, mesh: dict[str, np.ndarray], b_mats: np.ndarray, areas: np.ndarray, dof_map: np.ndarray) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    _, mu = elastic_constants(cfg)
    return {
        "nodes": torch.tensor(mesh["nodes"], dtype=dtype, device=device),
        "b_mats": torch.tensor(b_mats, dtype=dtype, device=device),
        "areas": torch.tensor(areas, dtype=dtype, device=device),
        "dof_map": torch.tensor(dof_map, dtype=torch.long, device=device),
        "cmat": torch.tensor(elastic_matrix_np(cfg), dtype=dtype, device=device),
        "mu": torch.tensor(mu, dtype=dtype, device=device),
        "hmod": torch.tensor(hardening_modulus_from_tangent(cfg), dtype=dtype, device=device),
        "kin_c1": torch.tensor(cfg.kin_c1, dtype=dtype, device=device),
        "kin_c2": torch.tensor(cfg.kin_c2, dtype=dtype, device=device),
    }


def build_free_dofs(mesh: dict[str, np.ndarray]) -> np.ndarray:
    fixed = set()
    for nid in mesh["left_nodes"]:
        fixed.add(2 * int(nid))
    for nid in mesh["bottom_nodes"]:
        fixed.add(2 * int(nid) + 1)
    for nid in mesh["top_nodes"]:
        fixed.add(2 * int(nid) + 1)
    return np.array([d for d in range(mesh["nodes"].shape[0] * 2) if d not in fixed], dtype=np.int64)


def isotropic_return_torch(
    cfg: Config,
    strain: torch.Tensor,
    eps_p_prev: torch.Tensor,
    p_prev: torch.Tensor,
    cmat: torch.Tensor,
    mu: torch.Tensor,
    hmod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trial = torch.matmul(strain - eps_p_prev, cmat.T)
    mean_trial = (trial[:, 0:1] + trial[:, 1:2] + trial[:, 2:3]) / 3.0
    s_trial = torch.cat([trial[:, 0:1] - mean_trial, trial[:, 1:2] - mean_trial, trial[:, 2:3] - mean_trial, trial[:, 3:4]], dim=1)
    vm_trial = torch.sqrt(torch.clamp(1.5 * (s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2 + s_trial[:, 2] ** 2 + 2.0 * s_trial[:, 3] ** 2), min=0.0))
    fy = vm_trial - (cfg.yield_stress + hmod * p_prev)
    elastic_mask = fy <= 0.0
    vm_safe = torch.clamp(vm_trial, min=1.0e-12)
    dgamma = torch.clamp(fy / (3.0 * mu + hmod), min=0.0)
    flow = 1.5 * s_trial / vm_safe.unsqueeze(1)
    eps_p_new = eps_p_prev + dgamma.unsqueeze(1) * flow
    p_new = p_prev + dgamma
    factor = 1.0 - 3.0 * mu * dgamma / vm_safe
    s_new = s_trial * factor.unsqueeze(1)
    stress_new = torch.cat([s_new[:, 0:1] + mean_trial, s_new[:, 1:2] + mean_trial, s_new[:, 2:3] + mean_trial, s_new[:, 3:4]], dim=1)
    stress = torch.where(elastic_mask.unsqueeze(1), trial, stress_new)
    eps_p = torch.where(elastic_mask.unsqueeze(1), eps_p_prev, eps_p_new)
    p_eq = torch.where(elastic_mask, p_prev, p_new)
    return stress, eps_p, p_eq


def mixed_hardening_update_torch(
    cfg: Config,
    strain: torch.Tensor,
    eps_p_prev: torch.Tensor,
    p_prev: torch.Tensor,
    x1_prev: torch.Tensor,
    x2_prev: torch.Tensor,
    cmat: torch.Tensor,
    mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trial = torch.matmul(strain - eps_p_prev, cmat.T)
    mean_trial = (trial[:, 0:1] + trial[:, 1:2] + trial[:, 2:3]) / 3.0
    s_trial = torch.cat([trial[:, 0:1] - mean_trial, trial[:, 1:2] - mean_trial, trial[:, 2:3] - mean_trial, trial[:, 3:4]], dim=1)
    x_old = x1_prev + x2_prev
    z_trial = s_trial - x_old
    z_norm = torch.sqrt(torch.clamp(1.5 * (z_trial[:, 0] ** 2 + z_trial[:, 1] ** 2 + z_trial[:, 2] ** 2 + 2.0 * z_trial[:, 3] ** 2), min=0.0))
    radius_prev = cfg.yield_stress + cfg.iso_q1 * (1.0 - torch.exp(-cfg.iso_b1 * p_prev))
    elastic_mask = z_norm <= radius_prev
    dpeq = torch.zeros_like(p_prev)
    for _ in range(30):
        p_new = p_prev + dpeq
        radius = cfg.yield_stress + cfg.iso_q1 * (1.0 - torch.exp(-cfg.iso_b1 * p_new))
        w1 = 1.0 / (1.0 + cfg.kin_gamma1 * dpeq)
        w2 = 1.0 / (1.0 + cfg.kin_gamma2 * dpeq)
        z = s_trial - w1.unsqueeze(1) * x1_prev - w2.unsqueeze(1) * x2_prev
        z_norm = torch.sqrt(torch.clamp(1.5 * (z[:, 0] ** 2 + z[:, 1] ** 2 + z[:, 2] ** 2 + 2.0 * z[:, 3] ** 2), min=1.0e-12))
        alpha = 1.0 + ((3.0 * mu) + w1 * cfg.kin_c1 + w2 * cfg.kin_c2) * dpeq / torch.clamp(radius, min=1.0e-12)
        residual = radius * alpha - z_norm
        new_dpeq = torch.clamp(dpeq - residual / torch.clamp((3.0 * mu) + cfg.kin_c1 + cfg.kin_c2 + abs(cfg.iso_q1 * cfg.iso_b1), min=1.0e-12), min=0.0)
        dpeq = torch.where(elastic_mask, torch.zeros_like(dpeq), new_dpeq)
    p_new = p_prev + dpeq
    radius = cfg.yield_stress + cfg.iso_q1 * (1.0 - torch.exp(-cfg.iso_b1 * p_new))
    w1 = 1.0 / (1.0 + cfg.kin_gamma1 * dpeq)
    w2 = 1.0 / (1.0 + cfg.kin_gamma2 * dpeq)
    z = s_trial - w1.unsqueeze(1) * x1_prev - w2.unsqueeze(1) * x2_prev
    z_norm = torch.sqrt(torch.clamp(1.5 * (z[:, 0] ** 2 + z[:, 1] ** 2 + z[:, 2] ** 2 + 2.0 * z[:, 3] ** 2), min=1.0e-12))
    nbar = 1.5 * z / z_norm.unsqueeze(1)
    alpha_scale = 1.0 + ((3.0 * mu) + w1 * cfg.kin_c1 + w2 * cfg.kin_c2) * dpeq / torch.clamp(radius, min=1.0e-12)
    x1_new = w1.unsqueeze(1) * (x1_prev + cfg.kin_c1 * dpeq.unsqueeze(1) * nbar)
    x2_new = w2.unsqueeze(1) * (x2_prev + cfg.kin_c2 * dpeq.unsqueeze(1) * nbar)
    s_new = x1_new + x2_new + z / alpha_scale.unsqueeze(1)
    stress_new = torch.cat([s_new[:, 0:1] + mean_trial, s_new[:, 1:2] + mean_trial, s_new[:, 2:3] + mean_trial, s_new[:, 3:4]], dim=1)
    eps_p_new = eps_p_prev + dpeq.unsqueeze(1) * nbar
    stress = torch.where(elastic_mask.unsqueeze(1), trial, stress_new)
    eps_p = torch.where(elastic_mask.unsqueeze(1), eps_p_prev, eps_p_new)
    p_eq = torch.where(elastic_mask, p_prev, p_new)
    x1 = torch.where(elastic_mask.unsqueeze(1), x1_prev, x1_new)
    x2 = torch.where(elastic_mask.unsqueeze(1), x2_prev, x2_new)
    return stress, eps_p, p_eq, x1, x2


def evaluate_state(
    cfg: Config,
    pred: torch.Tensor,
    data: dict[str, torch.Tensor],
    eps_p_prev: torch.Tensor,
    p_prev: torch.Tensor,
    x1_prev: torch.Tensor,
    x2_prev: torch.Tensor,
) -> dict[str, torch.Tensor]:
    u_flat = pred.reshape(-1)
    ue = u_flat[data["dof_map"]]
    strain = torch.einsum("eij,ej->ei", data["b_mats"], ue)
    if cfg.hardening_mode != "isotropic":
        raise ValueError("This path-dependent plate benchmark currently follows the paper's isotropic hardening setting.")
    stress, eps_p_new, p_new = isotropic_return_torch(cfg, strain, eps_p_prev, p_prev, data["cmat"], data["mu"], data["hmod"])
    x1_new = x1_prev
    x2_new = x2_prev
    fe = cfg.thickness * data["areas"].unsqueeze(1) * torch.einsum("eji,ej->ei", data["b_mats"], stress)
    fint = torch.zeros_like(u_flat)
    fint.index_add_(0, data["dof_map"].reshape(-1), fe.reshape(-1))
    eps_e = strain - eps_p_new
    delta_eps_p = eps_p_new - eps_p_prev
    delta_p = p_new - p_prev
    elastic_energy_density = 0.5 * voigt_inner(stress, eps_e)
    hardening_energy = 0.5 * data["hmod"] * p_new * p_new
    dissipation = voigt_inner(stress, delta_eps_p) - data["hmod"] * p_new * delta_p
    dem_energy_density = elastic_energy_density + hardening_energy + dissipation
    internal_energy = cfg.thickness * torch.sum(data["areas"] * dem_energy_density)
    return {
        "disp": u_flat,
        "residual": fint,
        "stress": stress,
        "eps_p": eps_p_new,
        "p_eq": p_new,
        "x1": x1_new,
        "x2": x2_new,
        "internal_energy": internal_energy,
        "external_work": torch.zeros((), dtype=u_flat.dtype, device=u_flat.device),
        "potential": internal_energy,
    }


def residual_metrics(residual: torch.Tensor, free_dofs: torch.Tensor) -> dict[str, torch.Tensor]:
    residual_free = residual[free_dofs]
    abs_res = torch.linalg.norm(residual_free)
    ref_norm = torch.clamp(torch.sqrt(torch.tensor(float(residual_free.numel()), dtype=residual.dtype, device=residual.device)), min=1.0)
    rel_res = abs_res / ref_norm
    return {"abs_res": abs_res, "ref_norm": ref_norm, "rel_res": rel_res, "loss_residual": rel_res * rel_res}


def train_feinn(
    cfg: Config,
    mesh: dict[str, np.ndarray],
    b_mats: np.ndarray,
    areas: np.ndarray,
    dof_map: np.ndarray,
    path_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, list[dict[str, float | int | np.ndarray]]]:
    tdata = build_torch_data(cfg, mesh, b_mats, areas, dof_map)
    free_dofs = torch.tensor(build_free_dofs(mesh), dtype=torch.long, device=tdata["nodes"].device)
    model = ResNet(2, 2, cfg.width_nn, cfg.blocks).to(dtype=getattr(torch, cfg.dtype), device=tdata["nodes"].device)
    x_in = normalize_coords(cfg, tdata["nodes"])
    ne = mesh["triangles"].shape[0]
    dtype = getattr(torch, cfg.dtype)
    device = tdata["nodes"].device
    eps_p_prev = torch.zeros((ne, 4), dtype=dtype, device=device)
    p_prev = torch.zeros(ne, dtype=dtype, device=device)
    x1_prev = torch.zeros((ne, 4), dtype=dtype, device=device)
    x2_prev = torch.zeros((ne, 4), dtype=dtype, device=device)
    history: list[dict[str, float | int]] = []
    response_rows: list[dict[str, float | int]] = []
    step_fields: list[dict[str, float | int | np.ndarray]] = []
    final_u = np.zeros(mesh["nodes"].shape[0] * 2, dtype=np.float64)
    final_state = {"stress": np.zeros((ne, 4), dtype=np.float64), "p_eq": np.zeros(ne, dtype=np.float64)}
    optimizer = torch.optim.Rprop(model.parameters(), lr=cfg.lr, etas=(0.5, 1.2), step_sizes=(1.0e-8, 10.0))
    e_scale = energy_scale(cfg)
    top_y_dofs = torch.tensor(2 * mesh["top_nodes"] + 1, dtype=torch.long, device=device)
    for row in path_df.itertuples(index=False):
        step = int(row.load_step)
        target_uy = float(row.top_uy)
        time_val = float(row.time)
        best_loss = math.inf
        best_state_dict = clone_model_state(model)
        dem_residual_scale = 0.0
        tick = time.time()
        print(f"[RPROP] path step {step:02d}/{len(path_df)} | t={time_val:.4f} | top_uy={target_uy:.6f}")
        for epoch in range(1, cfg.rprop_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in), target_uy)
            state = evaluate_state(cfg, pred, tdata, eps_p_prev, p_prev, x1_prev, x2_prev)
            metrics = residual_metrics(state["residual"], free_dofs)
            loss_potential = state["potential"] / e_scale
            loss_force = metrics["loss_residual"]
            if cfg.objective_mode == "dem":
                if dem_residual_scale == 0.0:
                    potential_scale = max(abs(float(loss_potential.detach().cpu())), 1.0e-12)
                    residual_scale = max(float(loss_force.detach().cpu()), 1.0e-12)
                    dem_residual_scale = potential_scale / residual_scale
                loss_objective = loss_potential + cfg.dem_residual_weight * dem_residual_scale * loss_force
            else:
                loss_objective = loss_force
            loss_reg = cfg.reg_weight * sum(torch.sum(p * p) for p in model.parameters())
            loss = loss_objective + loss_reg
            loss.backward()
            loss_scalar = float(loss.detach().cpu())
            if loss_scalar < best_loss:
                best_loss = loss_scalar
                best_state_dict = clone_model_state(model)
            optimizer.step()
            if epoch == 1 or epoch % 200 == 0 or epoch == cfg.rprop_epochs:
                elapsed = time.time() - tick
                print(f"[RPROP] step {step:02d} | epoch {epoch:5d}/{cfg.rprop_epochs} | loss={loss_scalar:.6e} | potential={float(loss_potential.detach().cpu()):.6e} | residual={float(loss_force.detach().cpu()):.6e} | dt={elapsed:.2f}s")
                history.append(
                    {
                        "load_step": step,
                        "time": time_val,
                        "top_uy": target_uy,
                        "epoch": epoch,
                        "loss_total": float(loss.detach().cpu()),
                        "loss_objective": float(loss_objective.detach().cpu()),
                        "loss_potential": float(loss_potential.detach().cpu()),
                        "loss_force": float(loss_force.detach().cpu()),
                        "internal_energy": float(state["internal_energy"].detach().cpu()),
                        "external_work": float(state["external_work"].detach().cpu()),
                        "abs_residual": float(metrics["abs_res"].detach().cpu()),
                        "rel_residual": float(metrics["rel_res"].detach().cpu()),
                        "elapsed_sec": elapsed,
                    }
                )
                tick = time.time()
        model.load_state_dict(best_state_dict)
        with torch.no_grad():
            pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in), target_uy)
            state = evaluate_state(cfg, pred, tdata, eps_p_prev, p_prev, x1_prev, x2_prev)
            metrics = residual_metrics(state["residual"], free_dofs)
            reaction_y = float(torch.sum(state["residual"][top_y_dofs]).detach().cpu())
            response_rows.append(
                {
                    "load_step": step,
                    "time": time_val,
                    "top_uy": target_uy,
                    "reaction_y": reaction_y,
                    "abs_residual": float(metrics["abs_res"].detach().cpu()),
                    "rel_residual": float(metrics["rel_res"].detach().cpu()),
                }
            )
            final_u = state["disp"].detach().cpu().numpy().reshape(-1)
            final_state = {"stress": state["stress"].detach().cpu().numpy(), "p_eq": state["p_eq"].detach().cpu().numpy()}
            snapshot_fields = postprocess(mesh, final_state, final_u)
            step_fields.append(
                {
                    "load_step": step,
                    "time": time_val,
                    "top_uy": target_uy,
                    **snapshot_fields,
                }
            )
            eps_p_prev = state["eps_p"].detach()
            p_prev = state["p_eq"].detach()
            x1_prev = state["x1"].detach()
            x2_prev = state["x2"].detach()
            print(f"[RPROP] path step {step:02d} done | reaction_y={reaction_y:.6e}")
    return final_u, final_state, pd.DataFrame(history), pd.DataFrame(response_rows), step_fields


def postprocess(mesh: dict[str, np.ndarray], state: dict[str, np.ndarray], displacement: np.ndarray) -> dict[str, np.ndarray]:
    nodes = mesh["nodes"]
    triangles = mesh["triangles"]
    stress_nodal = nodal_average(nodes.shape[0], triangles, state["stress"])
    peeq_elem = state["p_eq"][:, None]
    peeq_nodal = nodal_average(nodes.shape[0], triangles, peeq_elem).ravel()
    sigma_vm = von_mises_from_stress(stress_nodal)
    return {
        "ux": displacement[0::2],
        "uy": displacement[1::2],
        "sigma_xx": stress_nodal[:, 0],
        "sigma_yy": stress_nodal[:, 1],
        "sigma_xy": stress_nodal[:, 3],
        "sigma_vm": sigma_vm,
        "eps_p_eq": peeq_nodal,
    }


def save_fields_csv(nodes: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    pd.DataFrame(
        {
            "x": nodes[:, 0],
            "y": nodes[:, 1],
            "ux_fem": fem_fields["ux"],
            "uy_fem": fem_fields["uy"],
            "sigma_xx_fem": fem_fields["sigma_xx"],
            "sigma_yy_fem": fem_fields["sigma_yy"],
            "sigma_xy_fem": fem_fields["sigma_xy"],
            "sigma_vm_fem": fem_fields["sigma_vm"],
            "eps_p_eq_fem": fem_fields["eps_p_eq"],
            "ux_feinn": feinn_fields["ux"],
            "uy_feinn": feinn_fields["uy"],
            "sigma_xx_feinn": feinn_fields["sigma_xx"],
            "sigma_yy_feinn": feinn_fields["sigma_yy"],
            "sigma_xy_feinn": feinn_fields["sigma_xy"],
            "sigma_vm_feinn": feinn_fields["sigma_vm"],
            "eps_p_eq_feinn": feinn_fields["eps_p_eq"],
        }
    ).to_csv(out_path, index=False)


def key_step_indices(path_vals: np.ndarray) -> list[int]:
    n = len(path_vals)
    keep = {0, n - 1}
    if n <= 2:
        return sorted(keep)
    diffs = np.diff(path_vals)
    for i in range(1, n - 1):
        left = diffs[i - 1]
        right = diffs[i]
        if abs(left) < 1.0e-12 or abs(right) < 1.0e-12 or np.sign(left) != np.sign(right):
            keep.add(i)
    return sorted(keep)


def save_key_steps_fields_csv(
    nodes: np.ndarray,
    path_df: pd.DataFrame,
    step_fields: list[dict[str, float | int | np.ndarray]],
    out_path: Path,
) -> None:
    keep = set(key_step_indices(path_df["top_uy"].to_numpy(dtype=np.float64)))
    frames: list[pd.DataFrame] = []
    for idx, snap in enumerate(step_fields):
        if idx not in keep:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "load_step": int(snap["load_step"]),
                    "time": float(snap["time"]),
                    "top_uy": float(snap["top_uy"]),
                    "node_id": np.arange(1, nodes.shape[0] + 1, dtype=np.int64),
                    "x": nodes[:, 0],
                    "y": nodes[:, 1],
                    "ux": snap["ux"],
                    "uy": snap["uy"],
                    "sigma_xx": snap["sigma_xx"],
                    "sigma_yy": snap["sigma_yy"],
                    "sigma_xy": snap["sigma_xy"],
                    "sigma_vm": snap["sigma_vm"],
                    "eps_p_eq": snap["eps_p_eq"],
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)


def plot_panel(nodes: np.ndarray, triangles: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    tri_obj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    names = [("uy", r"$u_y$"), ("sigma_yy", r"$\sigma_{yy}$"), ("sigma_vm", r"$\sigma_{vm}$"), ("eps_p_eq", r"$\bar{\varepsilon}^p$")]
    fig, axes = plt.subplots(len(names), 3, figsize=(9.5, 11.5), dpi=200)
    for j, title in enumerate(["FEM", "FEINN", "Error"]):
        axes[0, j].set_title(title, fontsize=11)
    for i, (key, label) in enumerate(names):
        fem = fem_fields[key]
        pred = feinn_fields[key]
        err = pred - fem
        specs = [(fem, float(np.min(fem)), float(np.max(fem))), (pred, float(np.min(fem)), float(np.max(fem))), (err, float(np.min(err)), float(np.max(err)))]
        for j, (field, vmin, vmax) in enumerate(specs):
            ax = axes[i, j]
            im = ax.tripcolor(tri_obj, field, shading="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, rotation=0, labelpad=22, va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_path_response(fem_curve: pd.DataFrame, feinn_curve: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
    ax.plot(fem_curve["top_uy"], fem_curve["reaction_y"], marker="o", label="FEM")
    ax.plot(feinn_curve["top_uy"], feinn_curve["reaction_y"], marker="s", label="FEINN")
    ax.set_xlabel("Top displacement")
    ax.set_ylabel("Reaction Y")
    ax.set_title("Path Response")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], fem_curve: pd.DataFrame, feinn_curve: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in fem_fields:
        err = feinn_fields[key] - fem_fields[key]
        out[f"{key}_mae"] = float(np.mean(np.abs(err)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
    curve_err = feinn_curve["reaction_y"].to_numpy(dtype=np.float64) - fem_curve["reaction_y"].to_numpy(dtype=np.float64)
    out["reaction_curve_mae"] = float(np.mean(np.abs(curve_err)))
    out["reaction_curve_rmse"] = float(np.sqrt(np.mean(curve_err**2)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path-dependent FEINN for the perforated plate benchmark.")
    parser.add_argument("--fem-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--objective", type=str, choices=("dem", "dcm"), default="dem")
    parser.add_argument("--rprop-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--width-nn", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--dem-residual-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    fem_dir = Path(args.fem_dir) if args.fem_dir else base_dir / "outputs" / "path_perforated_plate_fem_case1"
    run_cfg = load_run_config(fem_dir / "path_perforated_plate_fem_run_config.txt")
    cfg = Config(
        fem_dir=str(fem_dir),
        objective_mode=args.objective,
        hardening_mode=str(run_cfg.get("hardening_mode", "kinematic")),
        path_case=str(run_cfg.get("path_case", "case1")),
        width=float(run_cfg.get("width", 200.0)),
        height=float(run_cfg.get("height", 200.0)),
        radius=float(run_cfg.get("radius", 50.0)),
        thickness=float(run_cfg.get("thickness", 100.0)),
        young=float(run_cfg.get("young", 7.0e4)),
        poisson=float(run_cfg.get("poisson", 0.20)),
        yield_stress=float(run_cfg.get("yield_stress", 250.0)),
        tangent_modulus=float(run_cfg.get("tangent_modulus", 2171.0)),
        iso_q1=float(run_cfg.get("iso_q1", -216.9135)),
        iso_b1=float(run_cfg.get("iso_b1", 213.9273)),
        kin_c1=float(run_cfg.get("kin_c1", 58791.656)),
        kin_gamma1=float(run_cfg.get("kin_gamma1", 147.7362)),
        kin_c2=float(run_cfg.get("kin_c2", 1803.7759)),
        kin_gamma2=float(run_cfg.get("kin_gamma2", 0.0)),
        load_steps=int(run_cfg.get("load_steps", 21)),
    )
    if cfg.hardening_mode != "isotropic":
        raise ValueError("Use FEM outputs generated with hardening_mode=isotropic for this benchmark.")
    if args.rprop_epochs is not None:
        cfg.rprop_epochs = args.rprop_epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.width_nn is not None:
        cfg.width_nn = args.width_nn
    if args.blocks is not None:
        cfg.blocks = args.blocks
    if args.dem_residual_weight is not None:
        cfg.dem_residual_weight = args.dem_residual_weight
    if args.device is not None:
        cfg.device = args.device

    set_seed(cfg.seed)
    objective_tag = objective_suffix(cfg.objective_mode)
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / f"path_perforated_plate_feinn_{cfg.path_case}_{objective_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Mesh] reading FEM exports from: {fem_dir}")
    mesh, fem_fields, path_df, fem_curve = load_fem_dataset(fem_dir)
    validate_path_data(cfg, path_df, fem_curve)
    print(f"[Path] using FEM-prescribed history from path_perforated_plate_fem_path.csv | case={cfg.path_case} | steps={len(path_df)}")
    nodes = mesh["nodes"]
    triangles = mesh["triangles"]
    b_mats, areas = build_tri_operators(nodes, triangles)
    dof_map = build_dof_map(triangles)

    print("[FEINN] training")
    feinn_u, feinn_state, history, response, step_fields = train_feinn(cfg, mesh, b_mats, areas, dof_map, path_df)
    feinn_fields = postprocess(mesh, feinn_state, feinn_u)

    print("[Post] exporting")
    save_fields_csv(nodes, fem_fields, feinn_fields, out_dir / "path_perforated_plate_feinn_fields.csv")
    save_key_steps_fields_csv(nodes, path_df, step_fields, out_dir / "path_perforated_plate_feinn_key_steps_fields.csv")
    history.to_csv(out_dir / "path_perforated_plate_feinn_training_history.csv", index=False)
    response.to_csv(out_dir / "path_perforated_plate_feinn_path_response.csv", index=False)
    plot_panel(nodes, triangles, fem_fields, feinn_fields, out_dir / "path_perforated_plate_feinn_panel.png")
    plot_path_response(fem_curve, response, out_dir / "path_perforated_plate_feinn_path_response.png")
    with open(out_dir / "path_perforated_plate_feinn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summarize_metrics(fem_fields, feinn_fields, fem_curve, response), f, indent=2)
    with open(out_dir / "path_perforated_plate_feinn_run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
