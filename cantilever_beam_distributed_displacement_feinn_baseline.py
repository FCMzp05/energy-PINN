#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


@dataclass
class Config:
    length: float = 60.0
    height: float = 20.0
    young: float = 200_000.0
    poisson: float = 0.3
    uy_right: float = -0.25
    nx: int = 20
    ny: int = 15
    adam_epochs: int = 2500
    lbfgs_steps: int = 300
    lr: float = 3.0e-3
    width: int = 64
    blocks: int = 4
    force_scale: float = 1.0e4
    objective_mode: str = "dcm"
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_mesh(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, cfg.length, cfg.nx + 1)
    ys = np.linspace(0.0, cfg.height, cfg.ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    elements = []
    for j in range(cfg.ny):
        for i in range(cfg.nx):
            n1 = j * (cfg.nx + 1) + i
            n2 = n1 + 1
            n4 = n1 + (cfg.nx + 1)
            n3 = n4 + 1
            elements.append([n1, n2, n3, n4])
    return nodes.astype(np.float64), np.array(elements, dtype=np.int64)


def constitutive_matrix(cfg: Config) -> np.ndarray:
    coef = cfg.young / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    lam = coef * cfg.poisson
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    return np.array(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )


def q4_shape_grads(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    dndxi = 0.25 * np.array(
        [
            -(1.0 - eta),
            +(1.0 - eta),
            +(1.0 + eta),
            -(1.0 + eta),
        ],
        dtype=np.float64,
    )
    dndeta = 0.25 * np.array(
        [
            -(1.0 - xi),
            -(1.0 + xi),
            +(1.0 + xi),
            +(1.0 - xi),
        ],
        dtype=np.float64,
    )
    return dndxi, dndeta


def build_element_operators(cfg: Config, nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gauss = 1.0 / math.sqrt(3.0)
    gauss_points = [(-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)]
    first_coords = nodes[elements[0]]

    b_mats = []
    detjs = []
    n_center = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    for xi, eta in gauss_points:
        dndxi, dndeta = q4_shape_grads(xi, eta)
        jac = np.array(
            [
                [np.dot(dndxi, first_coords[:, 0]), np.dot(dndeta, first_coords[:, 0])],
                [np.dot(dndxi, first_coords[:, 1]), np.dot(dndeta, first_coords[:, 1])],
            ],
            dtype=np.float64,
        )
        detj = float(np.linalg.det(jac))
        invj = np.linalg.inv(jac)
        grads = invj @ np.vstack([dndxi, dndeta])
        dndx = grads[0]
        dndy = grads[1]

        b = np.zeros((3, 8), dtype=np.float64)
        b[0, 0::2] = dndx
        b[1, 1::2] = dndy
        b[2, 0::2] = dndy
        b[2, 1::2] = dndx
        b_mats.append(b)
        detjs.append(detj)
    return np.stack(b_mats), np.array(detjs, dtype=np.float64), n_center


def build_dof_map(elements: np.ndarray) -> np.ndarray:
    dof_map = np.zeros((elements.shape[0], 8), dtype=np.int64)
    for e, conn in enumerate(elements):
        dofs = []
        for nid in conn:
            dofs.extend([2 * nid, 2 * nid + 1])
        dof_map[e] = np.array(dofs, dtype=np.int64)
    return dof_map


def boundary_sets(cfg: Config, nodes: np.ndarray) -> dict[str, np.ndarray]:
    tol = 1.0e-12
    left = np.where(np.isclose(nodes[:, 0], 0.0, atol=tol))[0]
    right = np.where(np.isclose(nodes[:, 0], cfg.length, atol=tol))[0]
    return {"left": left, "right": right}


def solve_fem(cfg: Config, nodes: np.ndarray, elements: np.ndarray, b_mats: np.ndarray, detjs: np.ndarray, dof_map: np.ndarray) -> np.ndarray:
    dmat = constitutive_matrix(cfg)
    ndof = nodes.shape[0] * 2
    stiff = np.zeros((ndof, ndof), dtype=np.float64)
    ke = np.zeros((8, 8), dtype=np.float64)
    for g in range(4):
        ke += b_mats[g].T @ dmat @ b_mats[g] * detjs[g]
    for e in range(elements.shape[0]):
        ids = dof_map[e]
        stiff[np.ix_(ids, ids)] += ke

    bc = {}
    bsets = boundary_sets(cfg, nodes)
    for nid in bsets["left"]:
        bc[2 * nid] = 0.0
        bc[2 * nid + 1] = 0.0
    for nid in bsets["right"]:
        bc[2 * nid + 1] = cfg.uy_right

    fixed = np.array(sorted(bc.keys()), dtype=np.int64)
    free = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)
    u = np.zeros(ndof, dtype=np.float64)
    for dof, value in bc.items():
        u[dof] = value

    rhs = -stiff[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(stiff[np.ix_(free, free)], rhs)
    return u


def average_element_fields_to_nodes(nodes: np.ndarray, elements: np.ndarray, element_values: np.ndarray) -> np.ndarray:
    nodal = np.zeros((nodes.shape[0], element_values.shape[1]), dtype=np.float64)
    counts = np.zeros(nodes.shape[0], dtype=np.float64)
    for e, conn in enumerate(elements):
        nodal[conn] += element_values[e]
        counts[conn] += 1.0
    nodal /= counts[:, None]
    return nodal


def postprocess_fields(cfg: Config, nodes: np.ndarray, elements: np.ndarray, dof_map: np.ndarray, b_mats: np.ndarray, displacement: np.ndarray) -> dict[str, np.ndarray]:
    dmat = constitutive_matrix(cfg)
    ue = displacement[dof_map]
    strain_gp = np.einsum("gij,ej->egi", b_mats, ue)
    stress_gp = np.einsum("ij,egj->egi", dmat, strain_gp)
    strain_el = strain_gp.mean(axis=1)
    stress_el = stress_gp.mean(axis=1)
    strain_nd = average_element_fields_to_nodes(nodes, elements, strain_el)
    stress_nd = average_element_fields_to_nodes(nodes, elements, stress_el)

    return {
        "ux": displacement[0::2],
        "uy": displacement[1::2],
        "ex": strain_nd[:, 0],
        "ey": strain_nd[:, 1],
        "sx": stress_nd[:, 0],
        "sy": stress_nd[:, 1],
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
    out[:, 0] = 2.0 * nodes_t[:, 0] / cfg.length - 1.0
    out[:, 1] = 2.0 * nodes_t[:, 1] / cfg.height - 1.0
    return out


def apply_hard_bc(cfg: Config, nodes_t: torch.Tensor, raw_out: torch.Tensor) -> torch.Tensor:
    xhat = nodes_t[:, 0:1] / cfg.length
    u_scale = 0.08
    v_scale = 0.08
    ux = xhat * u_scale * raw_out[:, 0:1]
    uy = xhat * (1.0 - xhat) * v_scale * raw_out[:, 1:2] + xhat * cfg.uy_right
    return torch.cat([ux, uy], dim=1)


def build_torch_data(cfg: Config, nodes: np.ndarray, elements: np.ndarray, b_mats: np.ndarray, detjs: np.ndarray, dof_map: np.ndarray) -> dict[str, torch.Tensor]:
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    return {
        "nodes": torch.tensor(nodes, dtype=dtype, device=device),
        "elements": torch.tensor(elements, dtype=torch.long, device=device),
        "b_mats": torch.tensor(b_mats, dtype=dtype, device=device),
        "detjs": torch.tensor(detjs, dtype=dtype, device=device),
        "dof_map": torch.tensor(dof_map, dtype=torch.long, device=device),
        "dmat": torch.tensor(constitutive_matrix(cfg), dtype=dtype, device=device),
    }


def build_free_dofs(cfg: Config, nodes: np.ndarray) -> np.ndarray:
    bsets = boundary_sets(cfg, nodes)
    constrained = set()
    for nid in bsets["left"]:
        constrained.add(2 * int(nid))
        constrained.add(2 * int(nid) + 1)
    for nid in bsets["right"]:
        constrained.add(2 * int(nid) + 1)
    return np.array([d for d in range(nodes.shape[0] * 2) if d not in constrained], dtype=np.int64)


def objective_suffix(mode: str) -> str:
    if mode not in {"dem", "dcm"}:
        raise ValueError(f"Unknown objective mode: {mode}")
    return mode


def energy_scale(cfg: Config) -> float:
    return max(cfg.young * cfg.height * abs(cfg.uy_right), 1.0)


def compute_internal_force(pred: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_flat = pred.reshape(-1)
    ue = u_flat[data["dof_map"]]
    strain_gp = torch.einsum("gij,ej->egi", data["b_mats"], ue)
    stress_gp = torch.einsum("ij,egj->egi", data["dmat"], strain_gp)
    fe = torch.einsum("gji,egj,g->ei", data["b_mats"], stress_gp, data["detjs"])

    fint = torch.zeros_like(u_flat)
    fint.index_add_(0, data["dof_map"].reshape(-1), fe.reshape(-1))
    internal_energy = 0.5 * torch.sum(torch.sum(strain_gp * stress_gp, dim=2) * data["detjs"])
    return fint, strain_gp, stress_gp, internal_energy


def train_feinn(cfg: Config, nodes: np.ndarray, elements: np.ndarray, b_mats: np.ndarray, detjs: np.ndarray, dof_map: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    tdata = build_torch_data(cfg, nodes, elements, b_mats, detjs, dof_map)
    free_dofs = torch.tensor(build_free_dofs(cfg, nodes), dtype=torch.long, device=tdata["nodes"].device)
    model = ResNet(2, 2, cfg.width, cfg.blocks).to(dtype=getattr(torch, cfg.dtype), device=tdata["nodes"].device)
    x_in = normalize_coords(cfg, tdata["nodes"])

    history: list[dict[str, float]] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start = time.time()
    print(f"[ADAM] start | epochs={cfg.adam_epochs} | lr={cfg.lr}")
    for epoch in range(1, cfg.adam_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in))
        fint, _, _, internal_energy = compute_internal_force(pred, tdata)
        loss_force = torch.mean((fint[free_dofs] / cfg.force_scale) ** 2)
        loss_potential = internal_energy / energy_scale(cfg)
        loss_objective = loss_potential if cfg.objective_mode == "dem" else loss_force
        loss_reg = 1.0e-8 * sum(torch.sum(p * p) for p in model.parameters())
        loss = loss_objective + loss_reg
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == cfg.adam_epochs:
            elapsed = time.time() - start
            print(
                f"[ADAM] epoch {epoch:5d}/{cfg.adam_epochs} | "
                f"loss={float(loss.detach().cpu()):.6e} | "
                f"objective={cfg.objective_mode} | "
                f"potential={float(loss_potential.detach().cpu()):.6e} | "
                f"force_loss={float(loss_force.detach().cpu()):.6e} | "
                f"dt={elapsed:.2f}s"
            )
            history.append(
                {
                    "epoch": epoch,
                    "stage": "adam",
                    "loss_total": float(loss.detach().cpu()),
                    "loss_objective": float(loss_objective.detach().cpu()),
                    "loss_potential": float(loss_potential.detach().cpu()),
                    "loss_force": float(loss_force.detach().cpu()),
                    "internal_energy": float(internal_energy.detach().cpu()),
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

    iter_counter = {"n": 0}
    print(f"[LBFGS] start | max_iter={cfg.lbfgs_steps}")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in))
        fint, _, _, internal_energy = compute_internal_force(pred, tdata)
        loss_force = torch.mean((fint[free_dofs] / cfg.force_scale) ** 2)
        loss_potential = internal_energy / energy_scale(cfg)
        loss_objective = loss_potential if cfg.objective_mode == "dem" else loss_force
        loss_reg = 1.0e-8 * sum(torch.sum(p * p) for p in model.parameters())
        loss = loss_objective + loss_reg
        loss.backward()
        iter_counter["n"] += 1
        if iter_counter["n"] == 1 or iter_counter["n"] % 25 == 0:
            print(
                f"[LBFGS] iter {iter_counter['n']:4d}/{cfg.lbfgs_steps} | "
                f"loss={float(loss.detach().cpu()):.6e} | "
                f"objective={cfg.objective_mode} | "
                f"potential={float(loss_potential.detach().cpu()):.6e} | "
                f"force_loss={float(loss_force.detach().cpu()):.6e}"
            )
            history.append(
                {
                    "epoch": iter_counter["n"],
                    "stage": "lbfgs",
                    "loss_total": float(loss.detach().cpu()),
                    "loss_objective": float(loss_objective.detach().cpu()),
                    "loss_potential": float(loss_potential.detach().cpu()),
                    "loss_force": float(loss_force.detach().cpu()),
                    "internal_energy": float(internal_energy.detach().cpu()),
                    "elapsed_sec": 0.0,
                }
            )
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        pred = apply_hard_bc(cfg, tdata["nodes"], model(x_in)).detach().cpu().numpy().reshape(-1)
    return pred, pd.DataFrame(history)


def plot_panel(cfg: Config, nodes: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    xx = nodes[:, 0].reshape(cfg.ny + 1, cfg.nx + 1)
    yy = nodes[:, 1].reshape(cfg.ny + 1, cfg.nx + 1)
    names = [
        ("ux", r"$u_x$ displacement"),
        ("uy", r"$u_y$ displacement"),
        ("ex", r"$\varepsilon_x$ strain"),
        ("ey", r"$\varepsilon_y$ strain"),
        ("sx", r"$\sigma_x$ stress"),
        ("sy", r"$\sigma_y$ stress"),
    ]

    fig, axes = plt.subplots(len(names), 3, figsize=(11, 15), dpi=220)
    col_titles = ["FEM", "FEINN", "Error"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    for i, (key, cbar_label) in enumerate(names):
        fem = fem_fields[key].reshape(cfg.ny + 1, cfg.nx + 1)
        pred = feinn_fields[key].reshape(cfg.ny + 1, cfg.nx + 1)
        err = pred - fem
        data_list = [fem, pred, err]
        cmaps = ["coolwarm", "coolwarm", "coolwarm"]
        for j, data in enumerate(data_list):
            ax = axes[i, j]
            im = ax.pcolormesh(xx, yy, data, shading="gouraud", cmap=cmaps[j])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if j == 0:
                ax.set_ylabel(key, rotation=0, labelpad=25, fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_fields_csv(nodes: np.ndarray, fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray], out_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": nodes[:, 0],
            "y": nodes[:, 1],
            "ux_fem": fem_fields["ux"],
            "uy_fem": fem_fields["uy"],
            "ex_fem": fem_fields["ex"],
            "ey_fem": fem_fields["ey"],
            "sx_fem": fem_fields["sx"],
            "sy_fem": fem_fields["sy"],
            "ux_feinn": feinn_fields["ux"],
            "uy_feinn": feinn_fields["uy"],
            "ex_feinn": feinn_fields["ex"],
            "ey_feinn": feinn_fields["ey"],
            "sx_feinn": feinn_fields["sx"],
            "sy_feinn": feinn_fields["sy"],
        }
    )
    frame.to_csv(out_path, index=False)


def summarize_metrics(fem_fields: dict[str, np.ndarray], feinn_fields: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for key in fem_fields:
        err = feinn_fields[key] - fem_fields[key]
        out[f"{key}_mae"] = float(np.mean(np.abs(err)))
        out[f"{key}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the FEINN elastic cantilever beam under distributed displacement.")
    parser.add_argument("--adam-epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--objective", type=str, default="dcm", choices=["dem", "dcm"])
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(adam_epochs=args.adam_epochs, lbfgs_steps=args.lbfgs_steps, objective_mode=args.objective)
    set_seed(cfg.seed)

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / f"cantilever_beam_distributed_displacement_feinn_baseline_{objective_suffix(cfg.objective_mode)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, elements = build_mesh(cfg)
    print(f"[Mesh] nodes={nodes.shape[0]} | elements={elements.shape[0]}")
    b_mats, detjs, _ = build_element_operators(cfg, nodes, elements)
    dof_map = build_dof_map(elements)

    print("[FEM] solving reference solution")
    fem_u = solve_fem(cfg, nodes, elements, b_mats, detjs, dof_map)
    fem_fields = postprocess_fields(cfg, nodes, elements, dof_map, b_mats, fem_u)

    print("[FEINN] training")
    feinn_u, history = train_feinn(cfg, nodes, elements, b_mats, detjs, dof_map)
    feinn_fields = postprocess_fields(cfg, nodes, elements, dof_map, b_mats, feinn_u)
    print("[Post] exporting fields, metrics, and figures")

    save_fields_csv(nodes, fem_fields, feinn_fields, out_dir / "cantilever_distributed_displacement_fields.csv")
    history.to_csv(out_dir / "cantilever_distributed_displacement_training_history.csv", index=False)
    plot_panel(cfg, nodes, fem_fields, feinn_fields, out_dir / "cantilever_distributed_displacement_panel.png")

    with open(out_dir / "cantilever_distributed_displacement_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summarize_metrics(fem_fields, feinn_fields), f, indent=2)
    with open(out_dir / "cantilever_distributed_displacement_run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Output directory: {out_dir}")
    print("Saved:")
    print("  cantilever_distributed_displacement_fields.csv")
    print("  cantilever_distributed_displacement_training_history.csv")
    print("  cantilever_distributed_displacement_panel.png")
    print("  cantilever_distributed_displacement_metrics.json")
    print("  cantilever_distributed_displacement_run_config.json")


if __name__ == "__main__":
    main()
