#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import matplotlib.tri as mtri


@dataclass
class Config:
    width: float = 0.6
    height: float = 0.6
    radius: float = 0.1
    thickness: float = 1.0
    traction_x: float = 1.0
    young: float = 20.0
    poisson: float = 0.25
    adam_epochs: int = 6000
    lbfgs_steps: int = 400
    lr: float = 5.0e-2
    optimizer_name: str = "rprop"
    width_nn: int = 96
    blocks: int = 6
    force_scale: float = 1.0
    reg_weight: float = 1.0e-10
    objective_mode: str = "dcm"
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"
    mesh_size_far: float = 0.06
    mesh_size_near: float = 0.015


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_gmsh_mesh(cfg: Config, msh_path: Path) -> dict[str, np.ndarray]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("defected_plate")

    p0 = gmsh.model.geo.addPoint(cfg.radius, 0.0, 0.0, cfg.mesh_size_near)
    p1 = gmsh.model.geo.addPoint(cfg.width, 0.0, 0.0, cfg.mesh_size_far)
    p2 = gmsh.model.geo.addPoint(cfg.width, cfg.height, 0.0, cfg.mesh_size_far)
    p3 = gmsh.model.geo.addPoint(0.0, cfg.height, 0.0, cfg.mesh_size_far)
    pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cfg.mesh_size_near)
    p4 = gmsh.model.geo.addPoint(0.0, cfg.radius, 0.0, cfg.mesh_size_near)

    l_bottom = gmsh.model.geo.addLine(p0, p1)
    l_right = gmsh.model.geo.addLine(p1, p2)
    l_top = gmsh.model.geo.addLine(p2, p3)
    l_left = gmsh.model.geo.addLine(p3, p4)
    l_arc = gmsh.model.geo.addCircleArc(p4, pc, p0)

    loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left, l_arc])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")
    gmsh.model.addPhysicalGroup(1, [l_left], 11)
    gmsh.model.setPhysicalName(1, 11, "left_sym")
    gmsh.model.addPhysicalGroup(1, [l_bottom], 12)
    gmsh.model.setPhysicalName(1, 12, "bottom_sym")
    gmsh.model.addPhysicalGroup(1, [l_right], 13)
    gmsh.model.setPhysicalName(1, 13, "right_traction")
    gmsh.model.addPhysicalGroup(1, [l_top], 14)
    gmsh.model.setPhysicalName(1, 14, "top_free")
    gmsh.model.addPhysicalGroup(1, [l_arc], 15)
    gmsh.model.setPhysicalName(1, 15, "hole")

    gmsh.model.mesh.generate(2)
    gmsh.write(str(msh_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

    tri_tags, tri_nodes = None, None
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 2:
            tri_nodes = np.array(enodes, dtype=np.int64).reshape(-1, 3)
            tri_tags = np.arange(tri_nodes.shape[0], dtype=np.int64)
            break
    if tri_nodes is None:
        gmsh.finalize()
        raise RuntimeError("No linear triangle elements found in gmsh mesh.")

    triangles = np.vectorize(tag_to_idx.get)(tri_nodes)

    right_edges = []
    _, _, edge_node_tags = gmsh.model.mesh.getElements(1, l_right)
    if edge_node_tags:
        right_line_nodes = np.array(edge_node_tags[0], dtype=np.int64).reshape(-1, 2)
        right_edges = np.vectorize(tag_to_idx.get)(right_line_nodes)
    right_edges = np.array(right_edges, dtype=np.int64)

    used_nodes = np.unique(triangles.reshape(-1))
    old_to_new = {int(old): i for i, old in enumerate(used_nodes)}
    nodes_used = node_coords[used_nodes]
    triangles = np.vectorize(old_to_new.get)(triangles)
    if right_edges.size > 0:
        right_edges = np.vectorize(old_to_new.get)(right_edges)

    gmsh.finalize()
    return {
        "nodes": nodes_used.astype(np.float64),
        "triangles": triangles.astype(np.int64),
        "right_edges": right_edges,
    }


def plane_stress_matrix(cfg: Config) -> np.ndarray:
    coef = cfg.young / (1.0 - cfg.poisson**2)
    return coef * np.array(
        [
            [1.0, cfg.poisson, 0.0],
            [cfg.poisson, 1.0, 0.0],
            [0.0, 0.0, (1.0 - cfg.poisson) / 2.0],
        ],
        dtype=np.float64,
    )


def build_tri_operators(cfg: Config, nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b_mats = []
    areas = []
    for tri in triangles:
        xy = nodes[tri]
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        fac = 1.0 / (2.0 * area)
        b = fac * np.array(
            [
                [b1, 0.0, b2, 0.0, b3, 0.0],
                [0.0, c1, 0.0, c2, 0.0, c3],
                [c1, b1, c2, b2, c3, b3],
            ],
            dtype=np.float64,
        )
        b_mats.append(b)
        areas.append(abs(area))
    return np.stack(b_mats), np.array(areas, dtype=np.float64)


def build_dof_map(triangles: np.ndarray) -> np.ndarray:
    out = np.zeros((triangles.shape[0], 6), dtype=np.int64)
    for e, tri in enumerate(triangles):
        dofs = []
        for nid in tri:
            dofs.extend([2 * nid, 2 * nid + 1])
        out[e] = np.array(dofs, dtype=np.int64)
    return out


def build_external_force(cfg: Config, nodes: np.ndarray, right_edges: np.ndarray) -> np.ndarray:
    ndof = nodes.shape[0] * 2
    fext = np.zeros(ndof, dtype=np.float64)
    traction = np.array([cfg.traction_x, 0.0], dtype=np.float64)
    for edge in right_edges:
        p1 = nodes[edge[0]]
        p2 = nodes[edge[1]]
        length = np.linalg.norm(p2 - p1)
        fe = cfg.thickness * length / 2.0 * np.array(
            [traction[0], traction[1], traction[0], traction[1]],
            dtype=np.float64,
        )
        edge_dofs = np.array([2 * edge[0], 2 * edge[0] + 1, 2 * edge[1], 2 * edge[1] + 1], dtype=np.int64)
        fext[edge_dofs] += fe
    return fext


def boundary_nodes(cfg: Config, nodes: np.ndarray) -> dict[str, np.ndarray]:
    tol = 1.0e-8
    left = np.where(np.isclose(nodes[:, 0], 0.0, atol=tol))[0]
    bottom = np.where(np.isclose(nodes[:, 1], 0.0, atol=tol))[0]
    return {"left": left, "bottom": bottom}


def solve_fem(cfg: Config, nodes: np.ndarray, triangles: np.ndarray, b_mats: np.ndarray, areas: np.ndarray, dof_map: np.ndarray, fext: np.ndarray) -> np.ndarray:
    dmat = plane_stress_matrix(cfg)
    ndof = nodes.shape[0] * 2
    stiff = np.zeros((ndof, ndof), dtype=np.float64)
    for e in range(triangles.shape[0]):
        ke = cfg.thickness * areas[e] * (b_mats[e].T @ dmat @ b_mats[e])
        ids = dof_map[e]
        stiff[np.ix_(ids, ids)] += ke

    bnodes = boundary_nodes(cfg, nodes)
    bc = {}
    for nid in bnodes["left"]:
        bc[2 * nid] = 0.0
    for nid in bnodes["bottom"]:
        bc[2 * nid + 1] = 0.0

    fixed = np.array(sorted(bc.keys()), dtype=np.int64)
    free = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)
    u = np.zeros(ndof, dtype=np.float64)
    for dof, value in bc.items():
        u[dof] = value

    rhs = fext[free] - stiff[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(stiff[np.ix_(free, free)], rhs)
    return u


def nodal_average(nodes: np.ndarray, triangles: np.ndarray, elem_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros((nodes.shape[0], elem_values.shape[1]), dtype=np.float64)
    counts = np.zeros(nodes.shape[0], dtype=np.float64)
    for e, tri in enumerate(triangles):
        nodal[tri] += elem_values[e]
        counts[tri] += 1.0
    nonzero = counts > 0.0
    nodal[nonzero] /= counts[nonzero, None]
    return nodal


def postprocess(cfg: Config, nodes: np.ndarray, triangles: np.ndarray, dof_map: np.ndarray, b_mats: np.ndarray, displacement: np.ndarray) -> dict[str, np.ndarray]:
    dmat = plane_stress_matrix(cfg)
    ue = displacement[dof_map]
    strain = np.einsum("eij,ej->ei", b_mats, ue)
    stress = np.einsum("ij,ej->ei", dmat, strain)
    stress_nodal = nodal_average(nodes, triangles, stress)
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
    yh = y / cfg.height

    # Enriched bubble-like envelopes keep the symmetry constraints exact
    # while giving the network more flexibility near the hole.
    bx = xh * (1.0 + 0.50 * yh + 0.25 * xh)
    by = yh * (1.0 + 0.50 * xh + 0.25 * yh)

    ux = cfg.width * bx * raw_out[:, 0:1]
    uy = cfg.height * by * raw_out[:, 1:2]
    return torch.cat([ux, uy], dim=1)


def build_torch_data(cfg: Config, nodes: np.ndarray, b_mats: np.ndarray, areas: np.ndarray, dof_map: np.ndarray, fext: np.ndarray) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    return {
        "nodes": torch.tensor(nodes, dtype=dtype, device=device),
        "b_mats": torch.tensor(b_mats, dtype=dtype, device=device),
        "areas": torch.tensor(areas, dtype=dtype, device=device),
        "dof_map": torch.tensor(dof_map, dtype=torch.long, device=device),
        "dmat": torch.tensor(plane_stress_matrix(cfg), dtype=dtype, device=device),
        "fext": torch.tensor(fext, dtype=dtype, device=device),
    }


def build_free_dofs(cfg: Config, nodes: np.ndarray) -> np.ndarray:
    bnodes = boundary_nodes(cfg, nodes)
    fixed = set()
    for nid in bnodes["left"]:
        fixed.add(2 * int(nid))
    for nid in bnodes["bottom"]:
        fixed.add(2 * int(nid) + 1)
    return np.array([d for d in range(nodes.shape[0] * 2) if d not in fixed], dtype=np.int64)


def objective_suffix(mode: str) -> str:
    if mode not in {"dem", "dcm"}:
        raise ValueError(f"Unknown objective mode: {mode}")
    return mode


def energy_scale(cfg: Config) -> float:
    return max(abs(cfg.traction_x) * cfg.width * cfg.height * cfg.thickness, 1.0)


def compute_internal_force(cfg: Config, pred: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_flat = pred.reshape(-1)
    ue = u_flat[data["dof_map"]]
    strain = torch.einsum("eij,ej->ei", data["b_mats"], ue)
    stress = torch.einsum("ij,ej->ei", data["dmat"], strain)
    fe = cfg.thickness * data["areas"].unsqueeze(1) * torch.einsum("eji,ej->ei", data["b_mats"], stress)
    fint = torch.zeros_like(u_flat)
    fint.index_add_(0, data["dof_map"].reshape(-1), fe.reshape(-1))
    internal_energy = 0.5 * cfg.thickness * torch.sum(torch.sum(strain * stress, dim=1) * data["areas"])
    return fint, strain, stress, internal_energy


def train_feinn(cfg: Config, nodes: np.ndarray, triangles: np.ndarray, b_mats: np.ndarray, areas: np.ndarray, dof_map: np.ndarray, fext: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    tdata = build_torch_data(cfg, nodes, b_mats, areas, dof_map, fext)
    free_dofs = torch.tensor(build_free_dofs(cfg, nodes), dtype=torch.long, device=tdata["nodes"].device)
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
        loss_reg = cfg.reg_weight * sum(torch.sum(p * p) for p in model.parameters())
        loss = loss_objective + loss_reg
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


def save_fields_csv(nodes: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    pd.DataFrame(
        {
            "x": nodes[:, 0],
            "y": nodes[:, 1],
            "ux_fem": fem_fields["ux"],
            "uy_fem": fem_fields["uy"],
            "sx_fem": fem_fields["sx"],
            "sy_fem": fem_fields["sy"],
            "sxy_fem": fem_fields["sxy"],
            "ux_feinn": feinn_fields["ux"],
            "uy_feinn": feinn_fields["uy"],
            "sx_feinn": feinn_fields["sx"],
            "sy_feinn": feinn_fields["sy"],
            "sxy_feinn": feinn_fields["sxy"],
        }
    ).to_csv(out_path, index=False)


def plot_panel(nodes: np.ndarray, triangles: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    tri_obj = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    names = [("ux", r"$u_x$"), ("uy", r"$u_y$"), ("sx", r"$\sigma_x$"), ("sy", r"$\sigma_y$"), ("sxy", r"$\sigma_{xy}$")]
    fig, axes = plt.subplots(len(names), 3, figsize=(10, 15), dpi=220)
    for j, title in enumerate(["FEM", "FEINN", "FEINN Error"]):
        axes[0, j].set_title(title, fontsize=12)
    for i, (key, label) in enumerate(names):
        fem = fem_fields[key]
        pred = feinn_fields[key]
        err = pred - fem
        for j, data in enumerate([fem, pred, err]):
            ax = axes[i, j]
            im = ax.tripcolor(tri_obj, data, shading="gouraud", cmap="jet")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
            ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
            if j == 0:
                ax.set_ylabel(label, rotation=0, labelpad=18, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for key in fem_fields:
        err = feinn_fields[key] - fem_fields[key]
        out[f"{key}_mae"] = float(np.mean(np.abs(err)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the FEINN elastic defected plate.")
    parser.add_argument("--objective", type=str, default="dcm", choices=["dem", "dcm"])
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(objective_mode=args.objective)
    set_seed(cfg.seed)

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / f"defected_plate_feinn_baseline_{objective_suffix(cfg.objective_mode)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "defected_plate.msh"

    print("[Mesh] generating gmsh mesh")
    mesh = make_gmsh_mesh(cfg, msh_path)
    nodes = mesh["nodes"]
    triangles = mesh["triangles"]
    right_edges = mesh["right_edges"]
    print(f"[Mesh] nodes={nodes.shape[0]} | triangles={triangles.shape[0]} | right_edges={right_edges.shape[0]}")

    b_mats, areas = build_tri_operators(cfg, nodes, triangles)
    dof_map = build_dof_map(triangles)
    fext = build_external_force(cfg, nodes, right_edges)

    print("[FEM] solving reference solution")
    fem_u = solve_fem(cfg, nodes, triangles, b_mats, areas, dof_map, fext)
    fem_fields = postprocess(cfg, nodes, triangles, dof_map, b_mats, fem_u)

    print("[FEINN] training")
    feinn_u, history = train_feinn(cfg, nodes, triangles, b_mats, areas, dof_map, fext)
    feinn_fields = postprocess(cfg, nodes, triangles, dof_map, b_mats, feinn_u)

    print("[Post] exporting fields, metrics, and figures")
    save_fields_csv(nodes, fem_fields, feinn_fields, out_dir / "defected_plate_fields.csv")
    history.to_csv(out_dir / "defected_plate_training_history.csv", index=False)
    plot_panel(nodes, triangles, fem_fields, feinn_fields, out_dir / "defected_plate_panel.png")
    with open(out_dir / "defected_plate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summarize_metrics(fem_fields, feinn_fields), f, indent=2)
    with open(out_dir / "defected_plate_run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Output directory: {out_dir}")
    print("Saved:")
    print("  defected_plate.msh")
    print("  defected_plate_fields.csv")
    print("  defected_plate_training_history.csv")
    print("  defected_plate_panel.png")
    print("  defected_plate_metrics.json")
    print("  defected_plate_run_config.json")


if __name__ == "__main__":
    main()
