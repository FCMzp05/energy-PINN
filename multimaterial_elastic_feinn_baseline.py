#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch


@dataclass
class Config:
    width: float = 20.0
    height: float = 10.0
    thickness: float = 1.0
    interface_y: float = 5.0
    young_top: float = 30.0
    poisson_top: float = 0.40
    young_bottom: float = 50.0
    poisson_bottom: float = 0.30
    traction_y: float = -2.0
    nx: int = 40
    ny: int = 20
    adam_epochs: int = 20000
    lbfgs_steps: int = 1200
    lr: float = 3.0e-1
    optimizer_name: str = "rprop"
    width_nn: int = 128
    blocks: int = 8
    force_scale: float = 40.0
    reg_weight: float = 1.0e-10
    hard_bc_gain: float = 4.0
    objective_mode: str = "dcm"
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_q4_mesh(cfg: Config) -> dict[str, np.ndarray]:
    xs = np.linspace(0.0, cfg.width, cfg.nx + 1)
    ys = np.linspace(0.0, cfg.height, cfg.ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    quads = []
    material_ids = []
    top_edges = []

    for j in range(cfg.ny):
        for i in range(cfg.nx):
            n0 = j * (cfg.nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (cfg.nx + 1)
            n2 = n3 + 1
            quad = [n0, n1, n2, n3]
            quads.append(quad)

            center_y = nodes[quad, 1].mean()
            material_ids.append(1 if center_y >= cfg.interface_y else 0)

            if j == cfg.ny - 1:
                top_edges.append([n3, n2])

    return {
        "nodes": nodes.astype(np.float64),
        "quads": np.array(quads, dtype=np.int64),
        "top_edges": np.array(top_edges, dtype=np.int64),
        "material_ids": np.array(material_ids, dtype=np.int64),
    }


def quads_to_tris(quads: np.ndarray) -> np.ndarray:
    tris = []
    for q in quads:
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
    return np.array(tris, dtype=np.int64)


def plane_strain_matrix(young: float, poisson: float) -> np.ndarray:
    coef = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return coef * np.array(
        [
            [1.0 - poisson, poisson, 0.0],
            [poisson, 1.0 - poisson, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * poisson) / 2.0],
        ],
        dtype=np.float64,
    )


def gauss_rule_q4() -> tuple[np.ndarray, np.ndarray]:
    a = 1.0 / np.sqrt(3.0)
    points = np.array([[-a, -a], [a, -a], [a, a], [-a, a]], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    return points, weights


def shape_grad_q4(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=np.float64,
    )


def build_q4_operators(nodes: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gpts, gweights = gauss_rule_q4()
    b_mats = np.zeros((quads.shape[0], 4, 3, 8), dtype=np.float64)
    detjw = np.zeros((quads.shape[0], 4), dtype=np.float64)

    for e, quad in enumerate(quads):
        xy = nodes[quad]
        for g, ((xi, eta), wg) in enumerate(zip(gpts, gweights)):
            dnds = shape_grad_q4(xi, eta)
            jac = dnds.T @ xy
            detj = np.linalg.det(jac)
            invj = np.linalg.inv(jac)
            dndx = dnds @ invj

            b = np.zeros((3, 8), dtype=np.float64)
            for a in range(4):
                b[0, 2 * a] = dndx[a, 0]
                b[1, 2 * a + 1] = dndx[a, 1]
                b[2, 2 * a] = dndx[a, 1]
                b[2, 2 * a + 1] = dndx[a, 0]

            b_mats[e, g] = b
            detjw[e, g] = detj * wg

    return b_mats, detjw


def build_dof_map(quads: np.ndarray) -> np.ndarray:
    out = np.zeros((quads.shape[0], 8), dtype=np.int64)
    for e, quad in enumerate(quads):
        dofs = []
        for nid in quad:
            dofs.extend([2 * nid, 2 * nid + 1])
        out[e] = np.array(dofs, dtype=np.int64)
    return out


def build_element_dmat(cfg: Config, material_ids: np.ndarray) -> np.ndarray:
    d_bottom = plane_strain_matrix(cfg.young_bottom, cfg.poisson_bottom)
    d_top = plane_strain_matrix(cfg.young_top, cfg.poisson_top)
    out = np.zeros((material_ids.shape[0], 3, 3), dtype=np.float64)
    out[material_ids == 0] = d_bottom
    out[material_ids == 1] = d_top
    return out


def build_external_force(cfg: Config, nodes: np.ndarray, top_edges: np.ndarray) -> np.ndarray:
    ndof = nodes.shape[0] * 2
    fext = np.zeros(ndof, dtype=np.float64)
    traction = np.array([0.0, cfg.traction_y], dtype=np.float64)
    gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    gw = np.ones(2, dtype=np.float64)

    for edge in top_edges:
        p1 = nodes[edge[0]]
        p2 = nodes[edge[1]]
        length = np.linalg.norm(p2 - p1)
        fe = np.zeros(4, dtype=np.float64)
        for s, w in zip(gp, gw):
            n = 0.5 * np.array([1.0 - s, 1.0 + s], dtype=np.float64)
            nmat = np.array(
                [
                    [n[0], 0.0, n[1], 0.0],
                    [0.0, n[0], 0.0, n[1]],
                ],
                dtype=np.float64,
            )
            fe += (nmat.T @ traction) * (length / 2.0) * w * cfg.thickness

        edge_dofs = np.array([2 * edge[0], 2 * edge[0] + 1, 2 * edge[1], 2 * edge[1] + 1], dtype=np.int64)
        fext[edge_dofs] += fe
    return fext


def boundary_data(nodes: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    tol = 1.0e-8
    left = np.where(np.isclose(nodes[:, 0], 0.0, atol=tol))[0]
    right = np.where(np.isclose(nodes[:, 0], cfg.width, atol=tol))[0]
    bottom = np.where(np.isclose(nodes[:, 1], 0.0, atol=tol))[0]
    return {"left": left, "right": right, "bottom": bottom}


def solve_fem(
    cfg: Config,
    nodes: np.ndarray,
    quads: np.ndarray,
    b_mats: np.ndarray,
    detjw: np.ndarray,
    dof_map: np.ndarray,
    dmat_e: np.ndarray,
    fext: np.ndarray,
) -> np.ndarray:
    ndof = nodes.shape[0] * 2
    stiff = np.zeros((ndof, ndof), dtype=np.float64)

    for e in range(quads.shape[0]):
        ke = np.zeros((8, 8), dtype=np.float64)
        for g in range(4):
            ke += cfg.thickness * (b_mats[e, g].T @ dmat_e[e] @ b_mats[e, g]) * detjw[e, g]
        ids = dof_map[e]
        stiff[np.ix_(ids, ids)] += ke

    binfo = boundary_data(nodes, cfg)
    bc = {}
    for nid in binfo["left"]:
        bc[2 * int(nid)] = 0.0
        bc[2 * int(nid) + 1] = 0.0
    for nid in binfo["right"]:
        bc[2 * int(nid)] = 0.0
        bc[2 * int(nid) + 1] = 0.0
    for nid in binfo["bottom"]:
        bc[2 * int(nid)] = 0.0
        bc[2 * int(nid) + 1] = 0.0

    fixed = np.array(sorted(bc.keys()), dtype=np.int64)
    free = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)
    u = np.zeros(ndof, dtype=np.float64)

    rhs = fext[free] - stiff[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(stiff[np.ix_(free, free)], rhs)
    return u


def nodal_average(nodes: np.ndarray, quads: np.ndarray, elem_values_gp: np.ndarray) -> np.ndarray:
    elem_values = elem_values_gp.mean(axis=1)
    nodal = np.zeros((nodes.shape[0], elem_values.shape[1]), dtype=np.float64)
    counts = np.zeros(nodes.shape[0], dtype=np.float64)
    for e, quad in enumerate(quads):
        nodal[quad] += elem_values[e]
        counts[quad] += 1.0
    nodal /= np.maximum(counts[:, None], 1.0)
    return nodal


def postprocess(
    cfg: Config,
    nodes: np.ndarray,
    quads: np.ndarray,
    dof_map: np.ndarray,
    b_mats: np.ndarray,
    dmat_e: np.ndarray,
    displacement: np.ndarray,
) -> dict[str, np.ndarray]:
    ue = displacement[dof_map]
    strain = np.einsum("egij,ej->egi", b_mats, ue)
    stress = np.einsum("eij,egj->egi", dmat_e, strain)
    stress_nodal = nodal_average(nodes, quads, stress)
    return {
        "ux": displacement[0::2],
        "uy": displacement[1::2],
        "sx": stress_nodal[:, 0],
        "sy": stress_nodal[:, 1],
        "sxy": stress_nodal[:, 2],
    }


class ResBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.fc1(x))
        y = self.fc2(y)
        return self.act(x + y)


class ResNet(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, blocks: int):
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


def apply_hard_bc(cfg: Config, nodes_t: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
    x = nodes_t[:, 0:1]
    y = nodes_t[:, 1:2]
    xh = x / cfg.width
    xh_r = 1.0 - xh
    yh = y / cfg.height

    # ux = 0 on left, right, and bottom
    bx = cfg.hard_bc_gain * xh * xh_r * yh
    ux = cfg.width * bx * raw_out[:, 0:1]

    # uy = 0 on left, right, and bottom
    by = cfg.hard_bc_gain * xh * xh_r * yh
    uy = cfg.height * by * raw_out[:, 1:2]
    return torch.cat([ux, uy], dim=1)


def build_torch_data(
    cfg: Config,
    nodes: np.ndarray,
    b_mats: np.ndarray,
    detjw: np.ndarray,
    dof_map: np.ndarray,
    dmat_e: np.ndarray,
    fext: np.ndarray,
) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    return {
        "nodes": torch.tensor(nodes, dtype=dtype, device=device),
        "b_mats": torch.tensor(b_mats, dtype=dtype, device=device),
        "detjw": torch.tensor(detjw, dtype=dtype, device=device),
        "dof_map": torch.tensor(dof_map, dtype=torch.long, device=device),
        "dmat_e": torch.tensor(dmat_e, dtype=dtype, device=device),
        "fext": torch.tensor(fext, dtype=dtype, device=device),
    }


def build_free_dofs(nodes: np.ndarray, cfg: Config) -> np.ndarray:
    binfo = boundary_data(nodes, cfg)
    fixed = set()
    for nid in binfo["left"]:
        fixed.add(2 * int(nid))
        fixed.add(2 * int(nid) + 1)
    for nid in binfo["right"]:
        fixed.add(2 * int(nid))
        fixed.add(2 * int(nid) + 1)
    for nid in binfo["bottom"]:
        fixed.add(2 * int(nid))
        fixed.add(2 * int(nid) + 1)
    return np.array([d for d in range(nodes.shape[0] * 2) if d not in fixed], dtype=np.int64)


def objective_suffix(mode: str) -> str:
    if mode not in {"dem", "dcm"}:
        raise ValueError(f"Unknown objective mode: {mode}")
    return mode


def energy_scale(cfg: Config) -> float:
    return max(abs(cfg.traction_y) * cfg.width * cfg.height * cfg.thickness, 1.0)


def compute_internal_force(cfg: Config, pred: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_flat = pred.reshape(-1)
    ue = u_flat[data["dof_map"]]
    strain = torch.einsum("egij,ej->egi", data["b_mats"], ue)
    stress = torch.einsum("eij,egj->egi", data["dmat_e"], strain)
    bt = data["b_mats"].transpose(2, 3)
    fe = cfg.thickness * torch.einsum("egij,egj,eg->ei", bt, stress, data["detjw"])
    fint = torch.zeros_like(u_flat)
    fint.index_add_(0, data["dof_map"].reshape(-1), fe.reshape(-1))
    internal_energy = 0.5 * cfg.thickness * torch.sum(torch.sum(strain * stress, dim=2) * data["detjw"])
    return fint, strain, stress, internal_energy


def train_feinn(
    cfg: Config,
    nodes: np.ndarray,
    quads: np.ndarray,
    b_mats: np.ndarray,
    detjw: np.ndarray,
    dof_map: np.ndarray,
    dmat_e: np.ndarray,
    fext: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    tdata = build_torch_data(cfg, nodes, b_mats, detjw, dof_map, dmat_e, fext)
    free_dofs = torch.tensor(build_free_dofs(nodes, cfg), dtype=torch.long, device=tdata["nodes"].device)
    model = ResNet(2, 2, cfg.width_nn, cfg.blocks).to(dtype=getattr(torch, cfg.dtype), device=tdata["nodes"].device)
    x_in = normalize_coords(cfg, tdata["nodes"])
    history = []

    if cfg.optimizer_name.lower() == "rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=cfg.lr, etas=(0.5, 1.2), step_sizes=(1.0e-8, 10.0))
        opt_label = "RPROP"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        opt_label = "ADAM"

    start = time.time()
    print(f"[{opt_label}] start | epochs={cfg.adam_epochs} | lr={cfg.lr}")
    for epoch in range(1, cfg.adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in))
        fint, _, _, internal_energy = compute_internal_force(cfg, pred, tdata)
        residual = fint - tdata["fext"]
        loss_force = torch.mean((residual[free_dofs] / cfg.force_scale) ** 2)
        external_work = torch.dot(tdata["fext"], pred.reshape(-1))
        loss_potential = (internal_energy - external_work) / energy_scale(cfg)
        loss_objective = loss_potential if cfg.objective_mode == "dem" else loss_force
        loss_reg = cfg.reg_weight * sum(torch.sum(p * p) for p in model.parameters())
        loss = loss_objective + loss_reg
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 200 == 0 or epoch == cfg.adam_epochs:
            elapsed = time.time() - start
            print(
                f"[{opt_label}] epoch {epoch:5d}/{cfg.adam_epochs} | "
                f"loss={float(loss.detach().cpu()):.6e} | objective={cfg.objective_mode} | "
                f"potential={float(loss_potential.detach().cpu()):.6e} | "
                f"force_loss={float(loss_force.detach().cpu()):.6e} | dt={elapsed:.2f}s"
            )
            history.append(
                {
                    "epoch": epoch,
                    "stage": cfg.optimizer_name.lower(),
                    "loss_total": float(loss.detach().cpu()),
                    "loss_objective": float(loss_objective.detach().cpu()),
                    "loss_potential": float(loss_potential.detach().cpu()),
                    "loss_force": float(loss_force.detach().cpu()),
                    "internal_energy": float(internal_energy.detach().cpu()),
                    "external_work": float(external_work.detach().cpu()),
                    "elapsed_sec": elapsed,
                }
            )
            start = time.time()

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=cfg.lbfgs_steps,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    counter = {"n": 0}
    print(f"[LBFGS] start | max_iter={cfg.lbfgs_steps}")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in))
        fint, _, _, internal_energy = compute_internal_force(cfg, pred, tdata)
        residual = fint - tdata["fext"]
        loss_force = torch.mean((residual[free_dofs] / cfg.force_scale) ** 2)
        external_work = torch.dot(tdata["fext"], pred.reshape(-1))
        loss_potential = (internal_energy - external_work) / energy_scale(cfg)
        loss_objective = loss_potential if cfg.objective_mode == "dem" else loss_force
        loss = loss_objective
        loss.backward()
        counter["n"] += 1
        if counter["n"] == 1 or counter["n"] % 50 == 0:
            print(
                f"[LBFGS] iter {counter['n']:4d}/{cfg.lbfgs_steps} | "
                f"loss={float(loss.detach().cpu()):.6e} | objective={cfg.objective_mode} | "
                f"potential={float(loss_potential.detach().cpu()):.6e} | "
                f"force_loss={float(loss_force.detach().cpu()):.6e}"
            )
            history.append(
                {
                    "epoch": counter["n"],
                    "stage": "lbfgs",
                    "loss_total": float(loss.detach().cpu()),
                    "loss_objective": float(loss_objective.detach().cpu()),
                    "loss_potential": float(loss_potential.detach().cpu()),
                    "loss_force": float(loss_force.detach().cpu()),
                    "internal_energy": float(internal_energy.detach().cpu()),
                    "external_work": float(external_work.detach().cpu()),
                    "elapsed_sec": 0.0,
                }
            )
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in)).detach().cpu().numpy().reshape(-1)
    return pred, pd.DataFrame(history)


def save_fields_csv(
    nodes: np.ndarray,
    material_ids: np.ndarray,
    quads: np.ndarray,
    fem_fields: dict[str, np.ndarray],
    feinn_fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    nodal_material = nodal_average(nodes, quads, np.repeat(material_ids[:, None, None], 4, axis=1).astype(np.float64)).reshape(-1)
    pd.DataFrame(
        {
            "x": nodes[:, 0],
            "y": nodes[:, 1],
            "material_id": nodal_material,
            "ux_fem": fem_fields["ux"],
            "uy_fem": fem_fields["uy"],
            "ux_feinn": feinn_fields["ux"],
            "uy_feinn": feinn_fields["uy"],
            "sx_fem": fem_fields["sx"],
            "sy_fem": fem_fields["sy"],
            "sxy_fem": fem_fields["sxy"],
            "sx_feinn": feinn_fields["sx"],
            "sy_feinn": feinn_fields["sy"],
            "sxy_feinn": feinn_fields["sxy"],
        }
    ).to_csv(out_path, index=False)


def plot_loss(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=220)
    ax.plot(history["epoch"], history["loss_total"], color="#e76f51", linewidth=1.5, label="Total loss")
    ax.plot(history["epoch"], history["loss_force"], color="#2a9d8f", linewidth=1.5, label="Force loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_interface(ax: plt.Axes, cfg: Config) -> None:
    ax.plot([0.0, cfg.width], [cfg.interface_y, cfg.interface_y], color="#444444", linewidth=0.7)


def plot_panel(
    cfg: Config,
    nodes: np.ndarray,
    quads: np.ndarray,
    fem_fields: dict[str, np.ndarray],
    feinn_fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    tri_obj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], quads_to_tris(quads))
    names = [("ux", r"$u_x$"), ("uy", r"$u_y$"), ("sx", r"$\sigma_x$"), ("sy", r"$\sigma_y$"), ("sxy", r"$\sigma_{xy}$")]
    fig, axes = plt.subplots(len(names), 3, figsize=(10, 14), dpi=220)
    for j, title in enumerate(["FEINN", "FEM", "FEINN Error"]):
        axes[0, j].set_title(title, fontsize=12)
    for i, (key, label) in enumerate(names):
        pred = feinn_fields[key]
        fem = fem_fields[key]
        err = pred - fem
        for j, data in enumerate([pred, fem, err]):
            ax = axes[i, j]
            im = ax.tripcolor(tri_obj, data, shading="gouraud", cmap="jet")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            draw_interface(ax, cfg)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0.0, cfg.width)
            ax.set_ylim(0.0, cfg.height)
            if j == 0:
                ax.set_ylabel(label, rotation=0, labelpad=16, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for key in ["ux", "uy", "sx", "sy", "sxy"]:
        err = feinn_fields[key] - fem_fields[key]
        out[f"{key}_mae"] = float(np.mean(np.abs(err)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def write_mesh_csv(nodes: np.ndarray, quads: np.ndarray, material_ids: np.ndarray, out_path: Path) -> None:
    rows = []
    for e, q in enumerate(quads):
        rows.append(
            {
                "element_id": e,
                "n1": int(q[0]),
                "n2": int(q[1]),
                "n3": int(q[2]),
                "n4": int(q[3]),
                "material_id": int(material_ids[e]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the FEINN multi-material elastic example.")
    parser.add_argument("--objective", type=str, default="dcm", choices=["dem", "dcm"])
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(objective_mode=args.objective)
    set_seed(cfg.seed)

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / f"multimaterial_elastic_feinn_baseline_{objective_suffix(cfg.objective_mode)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Mesh] generating Q4 mesh")
    mesh = make_q4_mesh(cfg)
    nodes = mesh["nodes"]
    quads = mesh["quads"]
    top_edges = mesh["top_edges"]
    material_ids = mesh["material_ids"]
    print(f"[Mesh] nodes={nodes.shape[0]} | quads={quads.shape[0]} | top_edges={top_edges.shape[0]}")

    b_mats, detjw = build_q4_operators(nodes, quads)
    dof_map = build_dof_map(quads)
    dmat_e = build_element_dmat(cfg, material_ids)
    fext = build_external_force(cfg, nodes, top_edges)

    print("[FEM] solving reference solution")
    fem_u = solve_fem(cfg, nodes, quads, b_mats, detjw, dof_map, dmat_e, fext)
    fem_fields = postprocess(cfg, nodes, quads, dof_map, b_mats, dmat_e, fem_u)

    print("[FEINN] training")
    feinn_u, history = train_feinn(cfg, nodes, quads, b_mats, detjw, dof_map, dmat_e, fext)
    feinn_fields = postprocess(cfg, nodes, quads, dof_map, b_mats, dmat_e, feinn_u)

    print("[Post] exporting fields, metrics, and figures")
    write_mesh_csv(nodes, quads, material_ids, out_dir / "multimaterial_elastic_mesh.csv")
    save_fields_csv(nodes, material_ids, quads, fem_fields, feinn_fields, out_dir / "multimaterial_elastic_fields.csv")
    history.to_csv(out_dir / "multimaterial_elastic_training_history.csv", index=False)
    plot_loss(history, out_dir / "multimaterial_elastic_loss.png")
    plot_panel(cfg, nodes, quads, fem_fields, feinn_fields, out_dir / "multimaterial_elastic_panel.png")
    with open(out_dir / "multimaterial_elastic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summarize_metrics(fem_fields, feinn_fields), f, indent=2)
    with open(out_dir / "multimaterial_elastic_run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Output directory: {out_dir}")
    print("Saved:")
    print("  multimaterial_elastic_mesh.csv")
    print("  multimaterial_elastic_fields.csv")
    print("  multimaterial_elastic_training_history.csv")
    print("  multimaterial_elastic_loss.png")
    print("  multimaterial_elastic_panel.png")
    print("  multimaterial_elastic_metrics.json")
    print("  multimaterial_elastic_run_config.json")


if __name__ == "__main__":
    main()
