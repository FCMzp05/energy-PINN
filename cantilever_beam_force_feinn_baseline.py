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
    objective_mode: str = "dem"
    length: float = 1.0
    height: float = 0.2
    young: float = 2.0e5
    poisson: float = 0.3
    yield_stress: float = 200.0
    hardening: float = 5.0e4
    load_start: float | None = 3.0
    load_end: float | None = 5.0
    load_case: str = "right_mid_point"
    right_uy_start: float | None = None
    right_uy_end: float | None = None
    load_steps: int = 20
    fem_nx: int = 63
    fem_ny: int = 13
    width_nn: int = 24
    blocks: int = 2
    rprop_epochs: int = 4000
    lbfgs_steps: int = 200
    lr: float = 5.0e-2
    reg_weight: float = 1.0e-10
    history_every: int = 200
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


def load_top_load(path: Path, ndof: int) -> np.ndarray:
    rows = pd.read_csv(path)
    order = np.argsort(rows["dof_id"].to_numpy(dtype=np.int64))
    values = rows.loc[order, "load_value"].to_numpy(dtype=np.float64)
    if values.size != ndof:
        raise ValueError(f"Top-load vector size mismatch: expected {ndof}, got {values.size}.")
    return values


def quads_to_triangles(elements: np.ndarray) -> np.ndarray:
    tris = np.zeros((elements.shape[0] * 2, 3), dtype=np.int64)
    for e, elem in enumerate(elements):
        tris[2 * e] = np.array([elem[0], elem[1], elem[2]], dtype=np.int64)
        tris[2 * e + 1] = np.array([elem[0], elem[2], elem[3]], dtype=np.int64)
    return tris


def load_fem_dataset(fem_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    field_rows = pd.read_csv(fem_dir / "cantilever_beam_force_fem_fields.csv")
    elem_rows = pd.read_csv(fem_dir / "cantilever_beam_force_fem_elements.csv")
    order = np.argsort(field_rows["node_id"].to_numpy(dtype=np.int64))
    nodes = field_rows.loc[order, ["x", "y"]].to_numpy(dtype=np.float64)
    elements = elem_rows[["n1", "n2", "n3", "n4"]].to_numpy(dtype=np.int64) - 1
    ndof = nodes.shape[0] * 2
    mesh = {
        "nodes": nodes,
        "elements": elements,
        "triangles": quads_to_triangles(elements),
        "left_nodes": load_node_ids(fem_dir / "cantilever_beam_force_fem_boundary_left.csv"),
        "right_nodes": load_node_ids(fem_dir / "cantilever_beam_force_fem_boundary_right.csv"),
        "top_nodes": load_node_ids(fem_dir / "cantilever_beam_force_fem_boundary_top.csv"),
        "bottom_nodes": load_node_ids(fem_dir / "cantilever_beam_force_fem_boundary_bottom.csv"),
    }
    top_load_path = fem_dir / "cantilever_beam_force_fem_top_load.csv"
    if top_load_path.exists():
        mesh["top_load"] = load_top_load(top_load_path, ndof)
    fem_fields = {
        "ux": field_rows.loc[order, "ux"].to_numpy(dtype=np.float64),
        "uy": field_rows.loc[order, "uy"].to_numpy(dtype=np.float64),
        "sigma_vm": field_rows.loc[order, "sigma_vm"].to_numpy(dtype=np.float64),
        "eps_p_eq": field_rows.loc[order, "eps_p_eq"].to_numpy(dtype=np.float64),
    }
    fem_history = pd.read_csv(fem_dir / "cantilever_beam_force_fem_history.csv")
    return mesh, fem_fields, fem_history


def elastic_constants(cfg: Config) -> tuple[float, float]:
    lam = cfg.young * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    return lam, mu


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


def q4_reference_matrices(cfg: Config) -> tuple[np.ndarray, float]:
    dx = cfg.length / cfg.fem_nx
    dy = cfg.height / cfg.fem_ny
    g = 1.0 / math.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    b_all = np.zeros((4, 4, 8), dtype=np.float64)
    for idx, (xi, eta) in enumerate(gauss):
        dndxi = 0.25 * np.array([-(1.0 - eta), +(1.0 - eta), +(1.0 + eta), -(1.0 + eta)], dtype=np.float64)
        dndeta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), +(1.0 + xi), +(1.0 - xi)], dtype=np.float64)
        dndx = 2.0 * dndxi / dx
        dndy = 2.0 * dndeta / dy
        b = np.zeros((4, 8), dtype=np.float64)
        for a in range(4):
            col = 2 * a
            b[0, col] = dndx[a]
            b[1, col + 1] = dndy[a]
            b[3, col] = 0.5 * dndy[a]
            b[3, col + 1] = 0.5 * dndx[a]
        b_all[idx] = b
    det_j = dx * dy / 4.0
    return b_all, det_j


def build_element_dof_map(elements: np.ndarray) -> np.ndarray:
    edof = np.zeros((elements.shape[0], 8), dtype=np.int64)
    for e, elem in enumerate(elements):
        dofs: list[int] = []
        for nid in elem:
            dofs.extend([2 * int(nid), 2 * int(nid) + 1])
        edof[e] = np.array(dofs, dtype=np.int64)
    return edof


def control_mode(cfg: Config) -> str:
    if cfg.right_uy_start is not None and cfg.right_uy_end is not None:
        return "displacement"
    return "force"


def build_free_dofs(mesh: dict[str, np.ndarray], mode: str) -> np.ndarray:
    fixed = set()
    for nid in mesh["left_nodes"]:
        fixed.add(2 * int(nid))
        fixed.add(2 * int(nid) + 1)
    if mode == "displacement":
        for nid in mesh["right_nodes"]:
            fixed.add(2 * int(nid) + 1)
    return np.array([d for d in range(mesh["nodes"].shape[0] * 2) if d not in fixed], dtype=np.int64)


def load_schedule(cfg: Config) -> np.ndarray:
    mode = control_mode(cfg)
    if mode == "displacement":
        assert cfg.right_uy_end is not None
        assert cfg.right_uy_start is not None
        if cfg.load_steps <= 1:
            return np.array([cfg.right_uy_end], dtype=np.float64)
        return np.linspace(cfg.right_uy_start, cfg.right_uy_end, cfg.load_steps, dtype=np.float64)
    assert cfg.load_end is not None
    assert cfg.load_start is not None
    if cfg.load_steps <= 1:
        return np.array([cfg.load_end], dtype=np.float64)
    return np.linspace(cfg.load_start, cfg.load_end, cfg.load_steps, dtype=np.float64)


def nodal_average(elements: np.ndarray, elem_values: np.ndarray, nnodes: int) -> np.ndarray:
    nodal = np.zeros((nnodes, elem_values.shape[1]), dtype=np.float64)
    counts = np.zeros(nnodes, dtype=np.float64)
    for e, elem in enumerate(elements):
        nodal[elem] += elem_values[e]
        counts[elem] += 1.0
    counts[counts == 0.0] = 1.0
    return nodal / counts[:, None]


def von_mises_from_stress(stress: np.ndarray) -> np.ndarray:
    sx = stress[:, 0]
    sy = stress[:, 1]
    sz = stress[:, 2]
    sxy = stress[:, 3]
    return np.sqrt(np.maximum(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * sxy**2, 0.0))


def l2_relative_error(ref: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(ref**2)))
    if denom < 1.0e-12:
        return 0.0
    return float(np.sqrt(np.sum((pred - ref) ** 2)) / denom)


def energy_scale(cfg: Config) -> float:
    return max(cfg.young * cfg.length * cfg.height, 1.0)


def objective_suffix(mode: str) -> str:
    if mode not in {"dem", "dcm"}:
        raise ValueError(f"Unknown objective mode: {mode}")
    return mode


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
    def __init__(self, width: int, blocks: int, in_dim: int = 10) -> None:
        super().__init__()
        self.in_layer = torch.nn.Linear(in_dim, width)
        self.blocks = torch.nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.out_layer = torch.nn.Linear(width, 2)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.in_layer(x))
        for block in self.blocks:
            y = block(y)
        return self.out_layer(y)


def normalize_coords(cfg: Config, nodes_t: torch.Tensor) -> torch.Tensor:
    out = nodes_t.clone()
    out[:, 0] = 2.0 * nodes_t[:, 0] / cfg.length - 1.0
    out[:, 1] = 2.0 * nodes_t[:, 1] / cfg.height - 1.0
    return out


def global_mode_features(cfg: Config, nodes_t: torch.Tensor) -> torch.Tensor:
    xh = 2.0 * nodes_t[:, 0:1] / cfg.length - 1.0
    yh = 2.0 * nodes_t[:, 1:2] / cfg.height - 1.0
    return torch.cat(
        [
            xh,
            yh,
            xh * yh,
            xh**2,
            yh**2,
            xh**3,
            yh**3,
            xh * (1.0 - yh**2),
            (1.0 - xh**2) * yh,
            (1.0 - xh**2) * (1.0 - yh**2),
        ],
        dim=1,
    )


def apply_hard_bc(cfg: Config, nodes_t: torch.Tensor, raw_out: torch.Tensor, target_value: float, mode: str) -> torch.Tensor:
    xh = nodes_t[:, 0:1] / cfg.length
    ux = cfg.length * xh * raw_out[:, 0:1]
    if mode == "displacement":
        bubble_x = xh * (1.0 - xh)
        uy = target_value * xh + cfg.height * bubble_x * raw_out[:, 1:2]
    else:
        uy = cfg.height * xh * raw_out[:, 1:2]
    return torch.cat([ux, uy], dim=1)


def build_torch_data(
    cfg: Config,
    mesh: dict[str, np.ndarray],
    b_mats: np.ndarray,
    det_j: float,
    edof: np.ndarray,
) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    _, mu = elastic_constants(cfg)
    return {
        "nodes": torch.tensor(mesh["nodes"], dtype=dtype, device=device),
        "elements": torch.tensor(mesh["elements"], dtype=torch.long, device=device),
        "b_mats": torch.tensor(b_mats, dtype=dtype, device=device),
        "det_j": torch.tensor(det_j, dtype=dtype, device=device),
        "edof": torch.tensor(edof, dtype=torch.long, device=device),
        "cmat": torch.tensor(elastic_matrix_np(cfg), dtype=dtype, device=device),
        "mu": torch.tensor(mu, dtype=dtype, device=device),
    }


def maybe_top_load_tensor(mesh: dict[str, np.ndarray], dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    top_load = mesh.get("top_load")
    if top_load is None:
        return None
    return torch.tensor(top_load, dtype=dtype, device=device)


def radial_return_torch(
    cfg: Config,
    strain: torch.Tensor,
    eps_p_prev: torch.Tensor,
    alpha_prev: torch.Tensor,
    cmat: torch.Tensor,
    mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trial = torch.matmul(strain - eps_p_prev, cmat.T)
    mean_trial = (trial[:, 0:1] + trial[:, 1:2] + trial[:, 2:3]) / 3.0
    s_trial = torch.cat(
        [
            trial[:, 0:1] - mean_trial,
            trial[:, 1:2] - mean_trial,
            trial[:, 2:3] - mean_trial,
            trial[:, 3:4],
        ],
        dim=1,
    )
    seq_trial = torch.sqrt(
        torch.clamp(
            1.5 * (s_trial[:, 0] ** 2 + s_trial[:, 1] ** 2 + s_trial[:, 2] ** 2 + 2.0 * s_trial[:, 3] ** 2),
            min=0.0,
        )
    )
    fy = seq_trial - (cfg.yield_stress + cfg.hardening * alpha_prev)
    elastic_mask = fy <= 0.0
    seq_safe = torch.clamp(seq_trial, min=1.0e-12)
    dgamma = torch.clamp(fy / (3.0 * mu + cfg.hardening), min=0.0)
    flow = 1.5 * s_trial / seq_safe.unsqueeze(1)
    eps_p_new = eps_p_prev + dgamma.unsqueeze(1) * flow
    alpha_new = alpha_prev + dgamma
    factor = 1.0 - 3.0 * mu * dgamma / seq_safe
    s_new = s_trial * factor.unsqueeze(1)
    stress_new = torch.cat(
        [
            s_new[:, 0:1] + mean_trial,
            s_new[:, 1:2] + mean_trial,
            s_new[:, 2:3] + mean_trial,
            s_new[:, 3:4],
        ],
        dim=1,
    )
    stress = torch.where(elastic_mask.unsqueeze(1), trial, stress_new)
    eps_p = torch.where(elastic_mask.unsqueeze(1), eps_p_prev, eps_p_new)
    alpha = torch.where(elastic_mask, alpha_prev, alpha_new)
    eps_e = strain - eps_p
    return stress, eps_p, alpha, eps_e


def evaluate_state(
    cfg: Config,
    pred: torch.Tensor,
    data: dict[str, torch.Tensor],
    top_load: torch.Tensor | None,
    eps_p_prev: torch.Tensor,
    alpha_prev: torch.Tensor,
    target_value: float,
    mode: str,
) -> dict[str, torch.Tensor]:
    def voigt_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum(a[..., :3] * b[..., :3], dim=-1) + 2.0 * a[..., 3] * b[..., 3]

    u_flat = pred.reshape(-1)
    ue = u_flat[data["edof"]]
    strain_gp = torch.einsum("gij,ej->egi", data["b_mats"], ue)
    ne, ngp, _ = strain_gp.shape
    strain_flat = strain_gp.reshape(-1, 4)
    eps_p_flat = eps_p_prev.reshape(-1, 4)
    alpha_flat = alpha_prev.reshape(-1)
    stress_flat, eps_p_new_flat, alpha_new_flat, eps_e_flat = radial_return_torch(
        cfg,
        strain_flat,
        eps_p_flat,
        alpha_flat,
        data["cmat"],
        data["mu"],
    )
    stress_gp = stress_flat.reshape(ne, ngp, 4)
    eps_p_new = eps_p_new_flat.reshape(ne, ngp, 4)
    alpha_new = alpha_new_flat.reshape(ne, ngp)
    eps_e_gp = eps_e_flat.reshape(ne, ngp, 4)
    delta_eps_p_gp = eps_p_new - eps_p_prev
    delta_alpha_gp = alpha_new - alpha_prev
    fe_gp = torch.einsum("gki,egk->egi", data["b_mats"], stress_gp)
    fe = torch.sum(fe_gp, dim=1) * data["det_j"]
    fint = torch.zeros_like(u_flat)
    fint.index_add_(0, data["edof"].reshape(-1), fe.reshape(-1))
    if mode == "force":
        if top_load is None:
            raise ValueError("Force-control mode requires top_load from FEM outputs.")
        fext = top_load * target_value
        residual = fint - fext
        external_work = torch.dot(fext, u_flat)
    else:
        residual = fint
        internal_energy = torch.zeros((), dtype=u_flat.dtype, device=u_flat.device)
        external_work = torch.zeros((), dtype=u_flat.dtype, device=u_flat.device)
    hardening_energy = 0.5 * cfg.hardening * alpha_new * alpha_new
    dissipation = voigt_inner(stress_gp, delta_eps_p_gp) - cfg.hardening * alpha_new * delta_alpha_gp
    dem_energy_density = 0.5 * voigt_inner(stress_gp, eps_e_gp) + hardening_energy + dissipation
    internal_energy = torch.sum(dem_energy_density) * data["det_j"]
    potential = internal_energy - external_work
    return {
        "disp": u_flat,
        "residual": residual,
        "stress_gp": stress_gp,
        "eps_p_gp": eps_p_new,
        "alpha_gp": alpha_new,
        "internal_energy": internal_energy,
        "external_work": external_work,
        "potential": potential,
    }


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def residual_metrics(
    residual: torch.Tensor,
    free_dofs: torch.Tensor,
    top_load: torch.Tensor | None,
    target_value: float,
    mode: str,
) -> dict[str, torch.Tensor]:
    residual_free = residual[free_dofs]
    abs_res = torch.linalg.norm(residual_free)
    if mode == "force":
        if top_load is None:
            raise ValueError("Force-control mode requires top_load from FEM outputs.")
        ref_vec = top_load[free_dofs] * target_value
        ref_norm = torch.clamp(torch.linalg.norm(ref_vec), min=1.0)
    else:
        ref_norm = torch.clamp(torch.sqrt(torch.tensor(float(residual_free.numel()), dtype=residual.dtype, device=residual.device)), min=1.0)
    rel_res = abs_res / ref_norm
    return {
        "abs_res": abs_res,
        "ref_norm": ref_norm,
        "rel_res": rel_res,
        "loss_residual": rel_res * rel_res,
    }


def train_model(
    cfg: Config,
    mesh: dict[str, np.ndarray],
    b_mats: np.ndarray,
    det_j: float,
    edof: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    tdata = build_torch_data(cfg, mesh, b_mats, det_j, edof)
    device = tdata["nodes"].device
    dtype = tdata["nodes"].dtype
    mode = control_mode(cfg)
    top_load = maybe_top_load_tensor(mesh, dtype, device)
    free_dofs = torch.tensor(build_free_dofs(mesh, mode), dtype=torch.long, device=device)
    x_in = global_mode_features(cfg, tdata["nodes"])
    model = ResNet(cfg.width_nn, cfg.blocks, in_dim=x_in.shape[1]).to(device=device, dtype=dtype)
    ne = mesh["elements"].shape[0]
    eps_p_prev = torch.zeros((ne, 4, 4), dtype=dtype, device=device)
    alpha_prev = torch.zeros((ne, 4), dtype=dtype, device=device)
    history: list[dict[str, float | int]] = []
    e_scale = energy_scale(cfg)
    final_u = np.zeros(mesh["nodes"].shape[0] * 2, dtype=np.float64)
    final_state = {
        "stress_gp": np.zeros((ne, 4, 4), dtype=np.float64),
        "alpha_gp": np.zeros((ne, 4), dtype=np.float64),
    }

    for step_id, target_value in enumerate(load_schedule(cfg), start=1):
        target_label = "right_uy" if mode == "displacement" else "load"
        print(
            f"[Train] step {step_id:02d}/{cfg.load_steps} | "
            f"{target_label}={target_value:.6e} | rprop_epochs={cfg.rprop_epochs} | lbfgs_steps={cfg.lbfgs_steps}"
        )
        best_loss = math.inf
        best_state_dict = clone_model_state(model)
        rprop_history_every = max(cfg.history_every, 1)
        lbfgs_history_every = max(cfg.history_every // 10, 20)
        optimizer = torch.optim.Rprop(model.parameters(), lr=cfg.lr, etas=(0.5, 1.2), step_sizes=(1.0e-8, 10.0))
        tick = time.time()

        for epoch in range(1, cfg.rprop_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in), float(target_value), mode)
            state = evaluate_state(cfg, pred, tdata, top_load, eps_p_prev, alpha_prev, float(target_value), mode)
            metrics = residual_metrics(state["residual"], free_dofs, top_load, float(target_value), mode)
            loss_potential = state["potential"] / e_scale
            loss_objective = loss_potential if cfg.objective_mode == "dem" else metrics["loss_residual"]
            loss_reg = cfg.reg_weight * sum(torch.sum(p * p) for p in model.parameters())
            loss = loss_objective + loss_reg
            loss.backward()
            loss_scalar = float(loss.detach().cpu())
            is_record_epoch = epoch == 1 or epoch % rprop_history_every == 0 or epoch == cfg.rprop_epochs
            if loss_scalar < best_loss:
                best_loss = loss_scalar
                best_state_dict = clone_model_state(model)
            optimizer.step()
            if is_record_epoch:
                elapsed = time.time() - tick
                potential_value = float(state["potential"].detach().cpu())
                print(
                    f"[RPROP] step {step_id:02d} | epoch {epoch:5d}/{cfg.rprop_epochs} | "
                    f"loss={loss_scalar:.6e} | objective={cfg.objective_mode} | "
                    f"rel_res={float(metrics['rel_res'].detach().cpu()):.6e} | "
                    f"potential={potential_value:.6e} | internal={float(state['internal_energy'].detach().cpu()):.6e} | dt={elapsed:.2f}s"
                )
                history.append(
                    {
                        "load_step": step_id,
                        "target_value": float(target_value),
                        "stage": "rprop",
                        "epoch": epoch,
                        "loss_total": loss_scalar,
                        "loss_objective": float(loss_objective.detach().cpu()),
                        "loss_potential": float(loss_potential.detach().cpu()),
                        "loss_residual": float(metrics["loss_residual"].detach().cpu()),
                        "loss_reg": float(loss_reg.detach().cpu()),
                        "potential": potential_value,
                        "internal_energy": float(state["internal_energy"].detach().cpu()),
                        "external_work": float(state["external_work"].detach().cpu()),
                        "abs_residual": float(metrics["abs_res"].detach().cpu()),
                        "rel_residual": float(metrics["rel_res"].detach().cpu()),
                        "elapsed_sec": elapsed,
                    }
                )
                tick = time.time()

        if cfg.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=cfg.lbfgs_steps,
                tolerance_grad=1.0e-12,
                tolerance_change=1.0e-12,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            closure_calls = {"n": 0}
            tick = time.time()

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                pred_inner = apply_hard_bc(cfg, tdata["nodes"], model(x_in), float(target_value), mode)
                state_inner = evaluate_state(
                    cfg,
                    pred_inner,
                    tdata,
                    top_load,
                    eps_p_prev,
                    alpha_prev,
                    float(target_value),
                    mode,
                )
                metrics_inner = residual_metrics(state_inner["residual"], free_dofs, top_load, float(target_value), mode)
                loss_potential_inner = state_inner["potential"] / e_scale
                loss_objective_inner = (
                    loss_potential_inner if cfg.objective_mode == "dem" else metrics_inner["loss_residual"]
                )
                loss_reg_inner = cfg.reg_weight * sum(torch.sum(p * p) for p in model.parameters())
                loss_inner = loss_objective_inner + loss_reg_inner
                loss_inner.backward()
                loss_scalar_inner = float(loss_inner.detach().cpu())
                next_call = closure_calls["n"] + 1
                is_record_iter = next_call == 1 or next_call % lbfgs_history_every == 0 or next_call == cfg.lbfgs_steps
                if loss_scalar_inner < nonlocal_best[0]:
                    nonlocal_best[0] = loss_scalar_inner
                    nonlocal_state[0] = clone_model_state(model)
                closure_calls["n"] += 1
                if is_record_iter:
                    elapsed = time.time() - tick
                    potential_value = float(state_inner["potential"].detach().cpu())
                    history.append(
                        {
                            "load_step": step_id,
                            "target_value": float(target_value),
                            "stage": "lbfgs",
                            "epoch": closure_calls["n"],
                            "loss_total": loss_scalar_inner,
                            "loss_objective": float(loss_objective_inner.detach().cpu()),
                            "loss_potential": float(loss_potential_inner.detach().cpu()),
                            "loss_residual": float(metrics_inner["loss_residual"].detach().cpu()),
                            "loss_reg": float(loss_reg_inner.detach().cpu()),
                            "potential": potential_value,
                            "internal_energy": float(state_inner["internal_energy"].detach().cpu()),
                            "external_work": float(state_inner["external_work"].detach().cpu()),
                            "abs_residual": float(metrics_inner["abs_res"].detach().cpu()),
                            "rel_residual": float(metrics_inner["rel_res"].detach().cpu()),
                            "elapsed_sec": elapsed,
                        }
                    )
                    print(
                        f"[LBFGS] step {step_id:02d} | closure {closure_calls['n']:4d} | "
                        f"loss={loss_scalar_inner:.6e} | rel_res={float(metrics_inner['rel_res'].detach().cpu()):.6e} | "
                        f"potential={potential_value:.6e}"
                    )
                return loss_inner

            nonlocal_best = [best_loss]
            nonlocal_state = [best_state_dict]
            lbfgs.step(closure)
            best_loss = nonlocal_best[0]
            best_state_dict = nonlocal_state[0]

        model.load_state_dict(best_state_dict)
        with torch.no_grad():
            pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in), float(target_value), mode)
            state = evaluate_state(cfg, pred, tdata, top_load, eps_p_prev, alpha_prev, float(target_value), mode)
            metrics = residual_metrics(state["residual"], free_dofs, top_load, float(target_value), mode)
            eps_p_prev = state["eps_p_gp"].detach()
            alpha_prev = state["alpha_gp"].detach()
            final_u = state["disp"].detach().cpu().numpy()
            final_state = {
                "stress_gp": state["stress_gp"].detach().cpu().numpy(),
                "alpha_gp": state["alpha_gp"].detach().cpu().numpy(),
            }
            free_res = float(metrics["abs_res"].detach().cpu())
            rel_res = float(metrics["rel_res"].detach().cpu())
            print(f"[Train] step {step_id:02d} done | free_res={free_res:.6e} | rel_res={rel_res:.6e}")

    return final_u, final_state, pd.DataFrame(history)


def postprocess_fields(
    mesh: dict[str, np.ndarray],
    displacement: np.ndarray,
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    nnodes = mesh["nodes"].shape[0]
    elem_stress = state["stress_gp"].mean(axis=1)
    elem_alpha = state["alpha_gp"].mean(axis=1)[:, None]
    nodal_stress = nodal_average(mesh["elements"], elem_stress, nnodes)
    nodal_alpha = nodal_average(mesh["elements"], elem_alpha, nnodes).ravel()
    return {
        "ux": displacement[0::2],
        "uy": displacement[1::2],
        "sigma_vm": von_mises_from_stress(nodal_stress),
        "eps_p_eq": nodal_alpha,
    }


def save_fields_csv(
    mesh: dict[str, np.ndarray],
    fem_fields: dict[str, np.ndarray],
    pred_fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    pd.DataFrame(
        {
            "x": mesh["nodes"][:, 0],
            "y": mesh["nodes"][:, 1],
            "ux_fem": fem_fields["ux"],
            "uy_fem": fem_fields["uy"],
            "sigma_vm_fem": fem_fields["sigma_vm"],
            "eps_p_eq_fem": fem_fields["eps_p_eq"],
            "ux_feinn": pred_fields["ux"],
            "uy_feinn": pred_fields["uy"],
            "sigma_vm_feinn": pred_fields["sigma_vm"],
            "eps_p_eq_feinn": pred_fields["eps_p_eq"],
        }
    ).to_csv(out_path, index=False)


def plot_fields(
    mesh: dict[str, np.ndarray],
    fem_fields: dict[str, np.ndarray],
    pred_fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    tri_obj = mtri.Triangulation(mesh["nodes"][:, 0], mesh["nodes"][:, 1], mesh["triangles"])
    names = [("ux", r"$u_x$"), ("uy", r"$u_y$"), ("sigma_vm", r"$\sigma_{vm}$"), ("eps_p_eq", r"$\bar{\varepsilon}^p$")]
    fig, axes = plt.subplots(len(names), 3, figsize=(10.4, 11.5), dpi=220)
    for j, title in enumerate(["FEM", "FEINN", "Error"]):
        axes[0, j].set_title(title, fontsize=12)
    for i, (key, label) in enumerate(names):
        fem = fem_fields[key]
        pred = pred_fields[key]
        err = pred - fem
        fem_vmin = float(np.min(fem))
        fem_vmax = float(np.max(fem))
        err_vmin = float(np.min(err))
        err_vmax = float(np.max(err))
        for j, (data, vmin, vmax) in enumerate(
            [(fem, fem_vmin, fem_vmax), (pred, fem_vmin, fem_vmax), (err, err_vmin, err_vmax)]
        ):
            ax = axes[i, j]
            im = ax.tripcolor(tri_obj, data, shading="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, rotation=0, labelpad=24, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(history: pd.DataFrame, out_path: Path) -> None:
    if history.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=220, constrained_layout=True)
    axes[0].plot(history.index + 1, history["loss_total"].to_numpy(dtype=np.float64), lw=1.0)
    axes[0].set_title("Total Objective")
    axes[0].set_xlabel("Record")
    axes[0].set_ylabel("Loss")
    axes[1].plot(history.index + 1, history["internal_energy"].to_numpy(dtype=np.float64), lw=1.0)
    axes[1].set_title("Internal Energy")
    axes[1].set_xlabel("Record")
    axes[1].set_ylabel("Value")
    axes[2].plot(history.index + 1, history["loss_residual"].to_numpy(dtype=np.float64), lw=1.0)
    axes[2].set_yscale("log")
    axes[2].set_title("Relative Equilibrium Residual Loss")
    axes[2].set_xlabel("Record")
    axes[2].set_ylabel("Loss")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_sampling_figure(mesh: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 1.8), dpi=220)
    ax.scatter(mesh["nodes"][:, 0], mesh["nodes"][:, 1], s=2.0, c="tab:blue", alpha=0.8)
    ax.scatter(mesh["nodes"][mesh["left_nodes"], 0], mesh["nodes"][mesh["left_nodes"], 1], s=5.0, c="tab:red")
    ax.set_title("Mesh Nodes")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(fem_fields: dict[str, np.ndarray], pred_fields: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("ux", "uy", "sigma_vm", "eps_p_eq"):
        err = pred_fields[key] - fem_fields[key]
        out[f"{key}_mae"] = float(np.mean(np.abs(err)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
        out[f"{key}_l2rel"] = l2_relative_error(fem_fields[key], pred_fields[key])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone cantilever-beam FEINN weak-form baseline.")
    parser.add_argument("--fem-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--objective", type=str, choices=("dem", "dcm"), default="dem")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--width-nn", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--rprop-epochs", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(__file__).resolve().parent
    case_outputs_dir = local_dir / "outputs"
    fem_dir = Path(args.fem_dir) if args.fem_dir else case_outputs_dir / "cantilever_beam_force_fem"
    out_dir = Path(args.output_dir) if args.output_dir else case_outputs_dir / f"cantilever_beam_force_feinn_baseline_{objective_suffix(args.objective)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_run_config(fem_dir / "cantilever_beam_force_fem_run_config.txt")
    cfg = Config(
        fem_dir=str(fem_dir),
        objective_mode=args.objective,
        length=float(run_cfg.get("length", 1.0)),
        height=float(run_cfg.get("height", 0.2)),
        young=float(run_cfg.get("young", 2.0e5)),
        poisson=float(run_cfg.get("poisson", 0.3)),
        yield_stress=float(run_cfg.get("yield_stress", 200.0)),
        hardening=float(run_cfg.get("hardening", 5.0e4)),
        load_start=float(run_cfg.get("load_start", 3.0)) if "load_start" in run_cfg else None,
        load_end=float(run_cfg.get("load_end", 5.0)) if "load_end" in run_cfg else None,
        load_case=str(run_cfg.get("load_case", "right_mid_point")),
        right_uy_start=float(run_cfg.get("right_uy_start")) if "right_uy_start" in run_cfg else None,
        right_uy_end=float(run_cfg.get("right_uy_end")) if "right_uy_end" in run_cfg else None,
        load_steps=int(run_cfg.get("load_steps", 20)),
        fem_nx=int(run_cfg.get("fem_nx", 63)),
        fem_ny=int(run_cfg.get("fem_ny", 13)),
    )
    if args.width is not None:
        cfg.width_nn = args.width
    if args.width_nn is not None:
        cfg.width_nn = args.width_nn
    if args.blocks is not None:
        cfg.blocks = args.blocks
    if args.rprop_epochs is not None:
        cfg.rprop_epochs = args.rprop_epochs
    if args.lbfgs_steps is not None:
        cfg.lbfgs_steps = args.lbfgs_steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device

    set_seed(cfg.seed)
    print(f"[FEM] reading weak-form FEM exports from: {fem_dir}")
    mesh, fem_fields, fem_history = load_fem_dataset(fem_dir)
    print(f"[Mesh] nodes={mesh['nodes'].shape[0]} | elements={mesh['elements'].shape[0]}")

    b_mats, det_j = q4_reference_matrices(cfg)
    edof = build_element_dof_map(mesh["elements"])
    print("[FEINN weak] training")
    disp, state, history = train_model(cfg, mesh, b_mats, det_j, edof)
    pred_fields = postprocess_fields(mesh, disp, state)

    save_fields_csv(mesh, fem_fields, pred_fields, out_dir / "cantilever_beam_force_feinn_baseline_fields.csv")
    plot_fields(mesh, fem_fields, pred_fields, out_dir / "cantilever_beam_force_feinn_baseline_fields.png")
    save_loss_curve(history, out_dir / "cantilever_beam_force_feinn_baseline_loss.png")
    save_sampling_figure(mesh, out_dir / "cantilever_beam_force_feinn_baseline_sampling.png")
    history.to_csv(out_dir / "cantilever_beam_force_feinn_baseline_training_history.csv", index=False)
    fem_history.to_csv(out_dir / "cantilever_beam_force_feinn_baseline_fem_history.csv", index=False)
    with (out_dir / "cantilever_beam_force_feinn_baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summarize_metrics(fem_fields, pred_fields), f, indent=2)
    with (out_dir / "cantilever_beam_force_feinn_baseline_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Output directory: {out_dir}")
    print("Saved:")
    print("  cantilever_beam_force_feinn_baseline_fields.csv")
    print("  cantilever_beam_force_feinn_baseline_fields.png")
    print("  cantilever_beam_force_feinn_baseline_loss.png")
    print("  cantilever_beam_force_feinn_baseline_sampling.png")
    print("  cantilever_beam_force_feinn_baseline_training_history.csv")
    print("  cantilever_beam_force_feinn_baseline_metrics.json")
    print("  cantilever_beam_force_feinn_baseline_run_config.json")


if __name__ == "__main__":
    main()
