"""Checkpointed SLURM workloads for the non-Cartesian validity audit.

All solves use the clean ``A^H A`` objective, deterministic operator
normalization, zero initialization, and stopping based on a recomputed true
normal-equation residual.  Full maps are saved; no display/metric mask is
applied during generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from mr_recon.covariance_apply import (
    COVARIANCE_APPLY_DESCRIPTION,
    apply_regularized_covariance,
)
from mr_recon.dtypes import complex_dtype, real_dtype
from mr_recon.fourier import sigpy_nufft
from mr_recon.linops import batching_params, sense_linop

# Evaluation order for Sigma v: "reordered" (default since 2026-09-06) or "legacy".
COVARIANCE_ORDER = os.environ.get("PICO_COVARIANCE_ORDER", "reordered")

try:
    from noncart_audit_core import (
        normalized_to_physical_variance,
        solve_cg_true_residual,
    )
except ModuleNotFoundError:  # imported as scripts.run_noncart_weekend_audit
    from scripts.noncart_audit_core import (
        normalized_to_physical_variance,
        solve_cg_true_residual,
    )


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments/noncartesian_phantom/data"
DEFAULT_OUT = (
    ROOT / "experiments/noncartesian_phantom/results/noncart_validity_audit"
)
SIGMA = 1e-2


@dataclass
class Problem:
    R: int
    device: torch.device
    image: torch.Tensor
    support: torch.Tensor
    ksp_ref: torch.Tensor
    ksp_acc: torch.Tensor
    A_ref: object
    A_acc: object
    max_eigen_ref: float
    max_eigen_acc: float


@dataclass
class System:
    A: object
    max_eigen: float
    lambda_normalized: float
    max_iter: int
    rtol: float
    true_residual_every: int

    @property
    def lambda_physical(self) -> float:
        return self.lambda_normalized * self.max_eigen

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.normal(x) / self.max_eigen + self.lambda_normalized * x

    def solve_image_rhs(
        self, rhs: torch.Tensor, *, max_iter: Optional[int] = None, **solver_kwargs
    ):
        return solve_cg_true_residual(
            self.normal,
            rhs,
            max_iter=self.max_iter if max_iter is None else int(max_iter),
            rtol=self.rtol,
            true_residual_every=self.true_residual_every,
            **solver_kwargs,
        )

    def reconstruct_noise(self, noise: torch.Tensor):
        rhs = self.A.adjoint(noise.type(complex_dtype)) / math.sqrt(self.max_eigen)
        return self.solve_image_rhs(rhs)

    def reconstruct_data(self, ksp: torch.Tensor, *, snapshot_iters: Iterable[int]):
        rhs = self.A.adjoint(ksp.type(complex_dtype)) / math.sqrt(self.max_eigen)
        return solve_cg_true_residual(
            self.normal,
            rhs,
            max_iter=max(snapshot_iters),
            rtol=0.0,
            true_residual_every=self.true_residual_every,
            snapshot_iters=snapshot_iters,
        )

    def covariance_apply(
        self,
        probe: torch.Tensor,
        *,
        max_iter: Optional[int] = None,
        order: str = COVARIANCE_ORDER,
    ):
        """Return ``Sigma v`` in normalized units (``1/max_eigen`` of physical).

        Evaluated as ``M^-2 (A^H A v) / max_eigen`` with a Krylov-recycled,
        certified second solve (``mr_recon.covariance_apply``).  ``order="legacy"``
        reproduces the literal two-zero-start product used before 2026-09-06.
        """
        cap = self.max_iter if max_iter is None else int(max_iter)

        def solve(rhs, **kwargs):
            return self.solve_image_rhs(rhs, max_iter=cap, **kwargs)

        second, (info1, info2) = apply_regularized_covariance(
            solve,
            self.A.normal,
            probe,
            1.0 / self.max_eigen,
            recycle=(order != "legacy"),
            order="legacy" if order == "legacy" else "reordered",
        )
        return second, info1, info2


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    os.replace(tmp, path)


def _save_npy(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(array))
    os.replace(tmp, path)


def _append_or_replace_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def estimate_max_eigen(A, seed: int = 0, iterations: int = 60) -> float:
    generator = torch.Generator(device=A.torch_dev).manual_seed(seed)
    x = torch.randn(
        A.ishape, dtype=complex_dtype, device=A.torch_dev, generator=generator
    )
    x = x / torch.linalg.norm(x)
    for _ in range(iterations):
        y = A.normal(x)
        x = y / torch.linalg.norm(y)
    rayleigh = torch.real(torch.sum(x.conj() * A.normal(x))).item()
    return float(rayleigh * 1.01)


def build_problem(R: int, device: torch.device) -> Problem:
    def load(name, dtype):
        return torch.tensor(np.load(DATA_DIR / name), dtype=dtype, device=device)

    image = load("img.npy", complex_dtype)
    mps = load("mps.npy", complex_dtype)
    ksp = load("ksp.npy", complex_dtype)
    dcf = load("dcf.npy", real_dtype)
    trj = load("trj.npy", real_dtype)
    im_size = tuple(image.shape[-2:])
    nufft = sigpy_nufft(im_size, width=4)
    bparams = batching_params(mps.shape[0])
    A_ref = sense_linop(
        im_size,
        trj,
        mps,
        dcf,
        nufft,
        bparams=bparams,
        use_toeplitz=True,
        weight_data_consistency=False,
    )
    A_acc = sense_linop(
        im_size,
        trj[:, ::R, :],
        mps,
        dcf[:, ::R],
        nufft,
        bparams=bparams,
        use_toeplitz=True,
        weight_data_consistency=False,
    )
    max_ref = estimate_max_eigen(A_ref, seed=1701)
    max_acc = estimate_max_eigen(A_acc, seed=1701)
    support = image.abs() > 0.05 * image.abs().max()
    return Problem(
        R=R,
        device=device,
        image=image,
        support=support,
        ksp_ref=ksp,
        ksp_acc=ksp[:, :, ::R],
        A_ref=A_ref,
        A_acc=A_acc,
        max_eigen_ref=max_ref,
        max_eigen_acc=max_acc,
    )


def make_systems(
    problem: Problem,
    lambda_ref: float,
    lambda_acc: float,
    max_iter: int,
    rtol: float,
    true_residual_every: int,
):
    common = dict(
        max_iter=max_iter,
        rtol=rtol,
        true_residual_every=true_residual_every,
    )
    return (
        System(
            problem.A_ref,
            problem.max_eigen_ref,
            lambda_ref,
            **common,
        ),
        System(
            problem.A_acc,
            problem.max_eigen_acc,
            lambda_acc,
            **common,
        ),
    )


def config_payload(problem: Problem, args, systems: Iterable[System]) -> dict:
    systems = list(systems)
    return {
        "R": problem.R,
        "device": str(problem.device),
        "operator": "clean unweighted A^H A",
        "covariance_apply": (
            COVARIANCE_APPLY_DESCRIPTION
            if COVARIANCE_ORDER != "legacy"
            else "legacy: M^-1 (A^H A (M^-1 v)) with two zero-initialized solves"
        ),
        "dcf_objective_weighting": False,
        "dcf_warm_start": False,
        "initialization": "zero",
        "stopping": "recomputed ||rhs-Hx||_2/||rhs||_2",
        "pmr_accumulation": ("mean and variance over R(ksp + n)"
                                if getattr(args, "pmr_estimator", "noise_only") == "replica"
                                else "sum of |R(noise)|^2"),
        "noise_mode": getattr(args, "pmr_estimator", "noise_only"),
        "sigma": SIGMA,
        "max_eigen_ref": problem.max_eigen_ref,
        "max_eigen_acc": problem.max_eigen_acc,
        "normalization_power_iterations": 60,
        "systems": [
            {
                "max_eigen": system.max_eigen,
                "lambda_normalized": system.lambda_normalized,
                "lambda_physical": system.lambda_physical,
                "max_iter": system.max_iter,
                "true_residual_rtol": system.rtol,
                "true_residual_every": system.true_residual_every,
            }
            for system in systems
        ],
        "arguments": vars(args),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _bootstrap_roi_ratio(
    ref_powers: np.ndarray,
    acc_powers: np.ndarray,
    R: int,
    *,
    seed: int,
    n_bootstrap: int = 500,
):
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        ref = ref_powers[rng.integers(0, len(ref_powers), len(ref_powers))].mean()
        acc = acc_powers[rng.integers(0, len(acc_powers), len(acc_powers))].mean()
        values[index] = math.sqrt(acc / max(ref, 1e-30)) / math.sqrt(R)
    return {
        "estimate": math.sqrt(acc_powers.mean() / ref_powers.mean()) / math.sqrt(R),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": n_bootstrap,
    }


def _reference_maps(
    problem: Problem,
    sum_ref: torch.Tensor,
    sum_acc: torch.Tensor,
    count: int,
):
    var_ref_norm_abs = (sum_ref / count).cpu().numpy()
    var_acc_norm_abs = (sum_acc / count).cpu().numpy()
    var_ref_phys_abs = normalized_to_physical_variance(
        var_ref_norm_abs, problem.max_eigen_ref
    )
    var_acc_phys_abs = normalized_to_physical_variance(
        var_acc_norm_abs, problem.max_eigen_acc
    )
    ratio_normalized = np.sqrt(
        np.maximum(var_acc_norm_abs, 0.0)
        / np.maximum(var_ref_norm_abs, 1e-30)
    ) / math.sqrt(problem.R)
    ratio_common_units = np.sqrt(
        np.maximum(var_acc_phys_abs, 0.0)
        / np.maximum(var_ref_phys_abs, 1e-30)
    ) / math.sqrt(problem.R)
    return {
        "var_ref_normalized_abs": var_ref_norm_abs,
        "var_acc_normalized_abs": var_acc_norm_abs,
        "var_ref_physical_abs": var_ref_phys_abs,
        "var_acc_physical_abs": var_acc_phys_abs,
        "sd_ref_physical_abs": np.sqrt(np.maximum(var_ref_phys_abs, 0.0)),
        "sd_acc_physical_abs": np.sqrt(np.maximum(var_acc_phys_abs, 0.0)),
        "ratio_normalized": ratio_normalized,
        "ratio_common_units": ratio_common_units,
    }


def _save_reference_maps(out: Path, prefix: str, maps: Dict[str, np.ndarray]) -> None:
    for name, array in maps.items():
        _save_npy(out / f"{prefix}_{name}.npy", array)


def run_reference(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem = build_problem(args.R, device)
    if getattr(args, "lambda_physical", None) is not None:
        lam_ref = args.lambda_physical / problem.max_eigen_ref
        lam_acc = args.lambda_physical / problem.max_eigen_acc
    else:
        lam_ref = lam_acc = args.lambda_normalized
    ref_system, acc_system = make_systems(
        problem,
        lam_ref,
        lam_acc,
        args.max_iter,
        args.rtol,
        args.true_residual_every,
    )
    out = Path(args.out_dir) / "reference" / f"R{args.R}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    _json_dump(out / "config.json", config_payload(problem, args, [ref_system, acc_system]))

    generator_ref = torch.Generator(device=device).manual_seed(args.seed * 1000 + 11)
    generator_acc = torch.Generator(device=device).manual_seed(args.seed * 1000 + 29)
    shape = problem.image.shape
    sums = {
        key: torch.zeros(shape, dtype=torch.float64, device=device)
        for key in (
            "ref",
            "acc",
            "ref_even",
            "ref_odd",
            "acc_even",
            "acc_odd",
        )
    }
    roi_ref: List[float] = []
    roi_acc: List[float] = []
    solver_rows: List[dict] = []
    checkpoint_rows: List[dict] = []
    start_n = 0
    state_path = out / "resume_state.npz"
    if args.resume and state_path.exists():
        state = np.load(state_path)
        start_n = int(state["n"])
        for key in sums:
            sums[key] = torch.tensor(state[f"sum_{key}"], dtype=torch.float64, device=device)
        roi_ref = state["roi_ref"].astype(float).tolist()
        roi_acc = state["roi_acc"].astype(float).tolist()
        generator_ref.set_state(torch.tensor(state["rng_ref"], dtype=torch.uint8))
        generator_acc.set_state(torch.tensor(state["rng_acc"], dtype=torch.uint8))
        print(f"[reference] resumed from N={start_n}")

    checkpoints = sorted(set(args.checkpoints + [args.N_max]))
    t0 = time.time()
    for index in range(start_n + 1, args.N_max + 1):
        noise_ref = SIGMA * torch.randn(
            problem.ksp_ref.shape,
            dtype=problem.ksp_ref.dtype,
            device=device,
            generator=generator_ref,
        )
        noise_acc = SIGMA * torch.randn(
            problem.ksp_acc.shape,
            dtype=problem.ksp_acc.dtype,
            device=device,
            generator=generator_acc,
        )
        x_ref, info_ref = ref_system.reconstruct_noise(noise_ref)
        x_acc, info_acc = acc_system.reconstruct_noise(noise_acc)
        power_ref = x_ref.abs().double().square()
        power_acc = x_acc.abs().double().square()
        sums["ref"] += power_ref
        sums["acc"] += power_acc
        parity = "even" if index % 2 == 0 else "odd"
        sums[f"ref_{parity}"] += power_ref
        sums[f"acc_{parity}"] += power_acc
        roi_ref.append(
            float((power_ref[problem.support] / problem.max_eigen_ref).mean().item())
        )
        roi_acc.append(
            float((power_acc[problem.support] / problem.max_eigen_acc).mean().item())
        )
        solver_rows.append(
            {
                "N": index,
                "ref_iterations": info_ref["iterations"],
                "ref_true_relative_residual": info_ref["true_relative_residual"],
                "ref_converged": info_ref["converged"],
                "acc_iterations": info_acc["iterations"],
                "acc_true_relative_residual": info_acc["true_relative_residual"],
                "acc_converged": info_acc["converged"],
            }
        )

        if index % args.checkpoint_every == 0 or index in checkpoints:
            tmp = state_path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    n=index,
                    **{f"sum_{key}": value.cpu().numpy() for key, value in sums.items()},
                    roi_ref=np.asarray(roi_ref),
                    roi_acc=np.asarray(roi_acc),
                    rng_ref=generator_ref.get_state().cpu().numpy(),
                    rng_acc=generator_acc.get_state().cpu().numpy(),
                )
            os.replace(tmp, state_path)
            _append_or_replace_csv(out / "solver_diagnostics.csv", solver_rows)

        if index in checkpoints:
            maps = _reference_maps(problem, sums["ref"], sums["acc"], index)
            prefix = f"pmr_seed{args.seed}_N{index}"
            _save_reference_maps(out, prefix, maps)
            n_even = index // 2
            n_odd = index - n_even
            even_maps = _reference_maps(
                problem, sums["ref_even"], sums["acc_even"], n_even
            )
            odd_maps = _reference_maps(
                problem, sums["ref_odd"], sums["acc_odd"], n_odd
            )
            _save_npy(
                out / f"{prefix}_split_even_ratio_common_units.npy",
                even_maps["ratio_common_units"],
            )
            _save_npy(
                out / f"{prefix}_split_odd_ratio_common_units.npy",
                odd_maps["ratio_common_units"],
            )
            support = problem.support.cpu().numpy()
            split_nrmse = float(
                np.linalg.norm(
                    (
                        even_maps["ratio_common_units"][support]
                        - odd_maps["ratio_common_units"][support]
                    ).astype(np.float64)
                )
                / np.linalg.norm(
                    odd_maps["ratio_common_units"][support].astype(np.float64)
                )
            )
            bootstrap = _bootstrap_roi_ratio(
                np.asarray(roi_ref),
                np.asarray(roi_acc),
                problem.R,
                seed=args.seed * 100000 + index,
            )
            row = {
                "N": index,
                "elapsed_s": time.time() - t0,
                "split_half_nrmse_common_units_support5": split_nrmse,
                "roi_ratio_common_units": bootstrap["estimate"],
                "roi_ratio_ci95_low": bootstrap["ci95_low"],
                "roi_ratio_ci95_high": bootstrap["ci95_high"],
                "ref_converged_fraction": float(
                    np.mean([r["ref_converged"] for r in solver_rows])
                ),
                "acc_converged_fraction": float(
                    np.mean([r["acc_converged"] for r in solver_rows])
                ),
            }
            checkpoint_rows.append(row)
            _append_or_replace_csv(out / "checkpoint_summary.csv", checkpoint_rows)
            print(
                f"[reference] R={args.R} seed={args.seed} N={index} "
                f"split={split_nrmse:.4g} ROI={bootstrap['estimate']:.5g} "
                f"elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )


def _accumulate_scaling_regime(
    problem: Problem,
    ref_system: System,
    acc_system_selected: System,
    acc_system_common: System,
    args,
    out: Path,
):
    generator_ref = torch.Generator(device=problem.device).manual_seed(args.seed + 101)
    generator_acc = torch.Generator(device=problem.device).manual_seed(args.seed + 303)
    sums = {
        key: torch.zeros(problem.image.shape, dtype=torch.float64, device=problem.device)
        for key in ("ref", "acc_selected", "acc_common")
    }
    rows = []
    checkpoints = sorted(set(args.checkpoints + [args.N]))
    t0 = time.time()
    for index in range(1, args.N + 1):
        noise_ref = SIGMA * torch.randn(
            problem.ksp_ref.shape,
            dtype=problem.ksp_ref.dtype,
            device=problem.device,
            generator=generator_ref,
        )
        noise_acc = SIGMA * torch.randn(
            problem.ksp_acc.shape,
            dtype=problem.ksp_acc.dtype,
            device=problem.device,
            generator=generator_acc,
        )
        x_ref, ir = ref_system.reconstruct_noise(noise_ref)
        x_selected, ia = acc_system_selected.reconstruct_noise(noise_acc)
        x_common, ic = acc_system_common.reconstruct_noise(noise_acc)
        sums["ref"] += x_ref.abs().double().square()
        sums["acc_selected"] += x_selected.abs().double().square()
        sums["acc_common"] += x_common.abs().double().square()
        if index in checkpoints:
            var_ref_phys = (sums["ref"] / index / problem.max_eigen_ref).cpu().numpy()
            var_acc_selected_phys = (
                sums["acc_selected"] / index / problem.max_eigen_acc
            ).cpu().numpy()
            var_acc_common_phys = (
                sums["acc_common"] / index / problem.max_eigen_acc
            ).cpu().numpy()
            for name, array in {
                "var_ref_physical_abs": var_ref_phys,
                "var_acc_selected_dimless_physical_abs": var_acc_selected_phys,
                "var_acc_common_lambda_physical_abs": var_acc_common_phys,
                "sd_ref_physical_abs": np.sqrt(np.maximum(var_ref_phys, 0)),
                "sd_acc_selected_dimless_physical_abs": np.sqrt(
                    np.maximum(var_acc_selected_phys, 0)
                ),
                "sd_acc_common_lambda_physical_abs": np.sqrt(
                    np.maximum(var_acc_common_phys, 0)
                ),
                "ratio_selected_dimless_common_units": np.sqrt(
                    np.maximum(var_acc_selected_phys, 0)
                    / np.maximum(var_ref_phys, 1e-30)
                )
                / math.sqrt(problem.R),
                "ratio_common_physical_lambda": np.sqrt(
                    np.maximum(var_acc_common_phys, 0)
                    / np.maximum(var_ref_phys, 1e-30)
                )
                / math.sqrt(problem.R),
            }.items():
                _save_npy(out / f"N{index}_{name}.npy", array)
            rows.append(
                {
                    "N": index,
                    "elapsed_s": time.time() - t0,
                    "ref_true_residual": ir["true_relative_residual"],
                    "acc_selected_true_residual": ia["true_relative_residual"],
                    "acc_common_true_residual": ic["true_relative_residual"],
                }
            )
            _append_or_replace_csv(out / "checkpoint_summary.csv", rows)
            print(f"[scaling] R={problem.R} N={index}", flush=True)


def run_scaling(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem = build_problem(args.R, device)
    lambda_physical = args.lambda_normalized * problem.max_eigen_ref
    lambda_acc_common = lambda_physical / problem.max_eigen_acc
    ref_system, acc_selected = make_systems(
        problem,
        args.lambda_normalized,
        args.lambda_normalized,
        args.max_iter,
        args.rtol,
        args.true_residual_every,
    )
    _, acc_common = make_systems(
        problem,
        args.lambda_normalized,
        lambda_acc_common,
        args.max_iter,
        args.rtol,
        args.true_residual_every,
    )
    out = Path(args.out_dir) / "scaling" / f"R{args.R}"
    out.mkdir(parents=True, exist_ok=True)
    payload = config_payload(problem, args, [ref_system, acc_selected, acc_common])
    payload["common_lambda_physical"] = lambda_physical
    payload["lambda_acc_normalized_for_common_physical_lambda"] = lambda_acc_common
    _json_dump(out / "config.json", payload)
    _accumulate_scaling_regime(
        problem, ref_system, acc_selected, acc_common, args, out
    )


def _aligned_image_nrmse(x: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor):
    xv = x[mask]
    rv = reference[mask]
    denominator = torch.sum((xv.conj() * xv).real).clamp(min=1e-30)
    alpha = torch.sum(xv.conj() * rv) / denominator
    return float(torch.linalg.norm(alpha * xv - rv) / torch.linalg.norm(rv))


def _pico_map(
    problem: Problem,
    lambda_normalized: float,
    cap: int,
    n_probes: int,
    seed: int,
    args,
):
    ref_system, acc_system = make_systems(
        problem,
        lambda_normalized,
        lambda_normalized,
        cap,
        0.0,
        args.true_residual_every,
    )
    generator = torch.Generator(device=problem.device).manual_seed(seed)
    diag_ref = torch.zeros(problem.image.shape, dtype=torch.float64, device=problem.device)
    diag_acc = torch.zeros_like(diag_ref)
    residuals = []
    for _ in range(n_probes):
        phase = 2 * torch.pi * torch.rand(
            problem.image.shape, device=problem.device, generator=generator
        )
        probe = torch.polar(torch.ones_like(phase), phase).type(complex_dtype)
        yr, i1r, i2r = ref_system.covariance_apply(probe, max_iter=cap)
        ya, i1a, i2a = acc_system.covariance_apply(probe, max_iter=cap)
        diag_ref += (probe.conj() * yr).real.double()
        diag_acc += (probe.conj() * ya).real.double()
        residuals.append(
            max(
                i1r["true_relative_residual"],
                i2r["true_relative_residual"],
                i1a["true_relative_residual"],
                i2a["true_relative_residual"],
            )
        )
    diag_ref /= n_probes
    diag_acc /= n_probes
    ratio = torch.sqrt(
        torch.clamp(diag_acc, min=0.0) / torch.clamp(diag_ref, min=1e-30)
    ) / math.sqrt(problem.R)
    return diag_ref, diag_acc, ratio, residuals


def run_lambda(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem = build_problem(2, device)
    out = Path(args.out_dir) / "lambda_stability" / "R2"
    out.mkdir(parents=True, exist_ok=True)
    selection = {
        "prespecified_before_compute": True,
        "stable_if_all": {
            "true_relative_normal_residual": "<= 1e-5",
            "solution_relative_change_from_previous_checkpoint": "<= 1e-3",
            "objective_relative_change_from_previous_checkpoint": "<= 1e-5",
            "variance_ratio_map_relative_change_support5": "<= 2e-2",
        },
        "conclusion_wording_if_lambda0_fails": (
            "did not reach a stable practical solution within the tested "
            "convergence-controlled budget"
        ),
    }
    _json_dump(out / "selection_criteria.json", selection)
    _json_dump(out / "config.json", config_payload(problem, args, []))

    signal_rows = []
    map_rows = []
    snapshots_by_lambda = {}
    for lam in args.lambdas:
        _, acc_system = make_systems(
            problem,
            lam,
            lam,
            max(args.iterations),
            0.0,
            args.true_residual_every,
        )
        _, info = acc_system.reconstruct_data(
            problem.ksp_acc, snapshot_iters=args.iterations
        )
        snapshots = info["snapshots"]
        snapshots_by_lambda[lam] = snapshots
        final_cap = max(snapshots)
        final = snapshots[final_cap]
        previous = None
        previous_objective = None
        for cap in args.iterations:
            if cap not in snapshots:
                continue
            x_norm = snapshots[cap]
            x_phys = x_norm / math.sqrt(problem.max_eigen_acc)
            rhs = problem.A_acc.adjoint(problem.ksp_acc) / math.sqrt(
                problem.max_eigen_acc
            )
            true_rel = float(
                torch.linalg.norm(acc_system.normal(x_norm) - rhs)
                / torch.linalg.norm(rhs)
            )
            data_residual = problem.A_acc.forward(x_phys) - problem.ksp_acc
            objective = float(
                0.5 * torch.sum(data_residual.abs().double().square())
                + 0.5
                * acc_system.lambda_physical
                * torch.sum(x_phys.abs().double().square())
            )
            solution_change = (
                float(torch.linalg.norm(x_norm - previous) / torch.linalg.norm(x_norm))
                if previous is not None
                else float("nan")
            )
            objective_change = (
                abs(objective - previous_objective) / max(abs(objective), 1e-30)
                if previous_objective is not None
                else float("nan")
            )
            signal_rows.append(
                {
                    "lambda_normalized": lam,
                    "lambda_physical": acc_system.lambda_physical,
                    "iteration": cap,
                    "true_relative_normal_residual": true_rel,
                    "objective_physical": objective,
                    "objective_relative_change": objective_change,
                    "solution_norm_normalized": float(torch.linalg.norm(x_norm)),
                    "solution_norm_physical": float(torch.linalg.norm(x_phys)),
                    "solution_relative_change": solution_change,
                    "solution_relative_difference_final": float(
                        torch.linalg.norm(x_norm - final) / torch.linalg.norm(final)
                    ),
                    "image_nrmse_support5_scale_aligned": _aligned_image_nrmse(
                        x_phys, problem.image, problem.support
                    ),
                }
            )
            _save_npy(
                out / f"recon_lambda{lam:.0e}_iter{cap}_physical.npy",
                x_phys.detach().cpu().numpy(),
            )
            previous = x_norm
            previous_objective = objective
        _append_or_replace_csv(out / "signal_convergence.csv", signal_rows)

        previous_ratio = None
        final_ratio = None
        per_cap = []
        for cap in args.iterations:
            dr, da, ratio, residuals = _pico_map(
                problem,
                lam,
                cap,
                args.n_probes,
                args.seed,
                args,
            )
            support = problem.support
            if previous_ratio is None:
                change = float("nan")
            else:
                change = float(
                    torch.linalg.norm((ratio - previous_ratio)[support].float())
                    / torch.linalg.norm(ratio[support].float())
                )
            per_cap.append((cap, ratio.detach().clone()))
            map_rows.append(
                {
                    "lambda_normalized": lam,
                    "iteration": cap,
                    "n_probes": args.n_probes,
                    "variance_ratio_map_relative_change_support5": change,
                    "maximum_probe_true_relative_residual": max(residuals),
                }
            )
            _save_npy(
                out / f"pico_ratio_lambda{lam:.0e}_iter{cap}.npy",
                ratio.cpu().numpy(),
            )
            _save_npy(
                out / f"pico_diag_ref_lambda{lam:.0e}_iter{cap}.npy",
                dr.cpu().numpy(),
            )
            _save_npy(
                out / f"pico_diag_acc_lambda{lam:.0e}_iter{cap}.npy",
                da.cpu().numpy(),
            )
            previous_ratio = ratio
            final_ratio = ratio
            _append_or_replace_csv(out / "variance_map_convergence.csv", map_rows)
        for cap, ratio in per_cap:
            for row in reversed(map_rows):
                if row["lambda_normalized"] == lam and row["iteration"] == cap:
                    row["variance_ratio_map_relative_difference_final_support5"] = float(
                        torch.linalg.norm((ratio - final_ratio)[problem.support].float())
                        / torch.linalg.norm(final_ratio[problem.support].float())
                    )
                    break
        _append_or_replace_csv(out / "variance_map_convergence.csv", map_rows)
        print(f"[lambda] completed lambda={lam:g}", flush=True)


def _synced_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def run_matched(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem = build_problem(args.R, device)
    if getattr(args, "lambda_physical", None) is not None:
        lam_ref = args.lambda_physical / problem.max_eigen_ref
        lam_acc = args.lambda_physical / problem.max_eigen_acc
    else:
        lam_ref = lam_acc = args.lambda_normalized
    ref_system, acc_system = make_systems(
        problem,
        lam_ref,
        lam_acc,
        args.max_iter,
        args.rtol,
        args.true_residual_every,
    )
    out = (
        Path(args.out_dir)
        / "matched_incremental"
        / f"R{args.R}"
        / f"seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    _json_dump(out / "config.json", config_payload(problem, args, [ref_system, acc_system]))
    checkpoints = sorted(set(args.checkpoints + [args.N_max]))
    progress_rows = []

    if "pmr" in args.methods:
        method_start = _synced_time()
        # PICO applies one probe to both arms, so its two arms' errors are correlated
        # and partly cancel in the ratio sigma_acc/(sqrt(R) sigma_ref). --pmr-common-noise
        # gives PMR the same common-random-numbers treatment, which is what a like-for-like
        # comparison of the two estimators on the ratio map requires.
        common_noise = bool(getattr(args, "pmr_common_noise", False))
        generator_ref = torch.Generator(device=device).manual_seed(args.seed + 401)
        generator_acc = (generator_ref if common_noise
                         else torch.Generator(device=device).manual_seed(args.seed + 509))
        sum_ref = torch.zeros(problem.image.shape, dtype=torch.float64, device=device)
        sum_acc = torch.zeros_like(sum_ref)
        # "replica" is PMR as defined by Robson et al.: reconstruct the noisy data
        # ksp + n and take the voxelwise mean and variance across replicas (Welford,
        # accumulated in float64 over complex64 reconstructions).  "noise_only"
        # accumulates E|R(n)|^2 instead, which is the same quantity for a linear R
        # and avoids subtracting the reconstructed signal.
        replica = getattr(args, "pmr_estimator", "noise_only") == "replica"
        # Replicas are reconstructions of ksp + n, so the injected noise level sets how
        # well the solver and complex64 arithmetic can resolve the noise component: with
        # a relative-residual stop, a signal-dominated right-hand side converges long
        # before the noise part does.  The normalized ratio map is exactly independent of
        # the noise level for a linear reconstruction, so the scale below only fixes the
        # numerics; the stored maps are divided by it to stay in units of SIGMA^2.
        sigma_scale = float(getattr(args, "pmr_sigma_scale", 1.0))
        mean_ref = torch.zeros(problem.image.shape, dtype=torch.complex128, device=device)
        mean_acc = torch.zeros_like(mean_ref)
        for index in range(1, args.N_max + 1):
            nr = (SIGMA * sigma_scale) * torch.randn(
                problem.ksp_ref.shape,
                dtype=problem.ksp_ref.dtype,
                device=device,
                generator=generator_ref,
            )
            if common_noise:
                # the accelerated arm keeps every R-th shot of the same realization
                na = nr[:, :, ::problem.R] if nr.dim() == 3 else nr[..., ::problem.R]
            else:
                na = (SIGMA * sigma_scale) * torch.randn(
                    problem.ksp_acc.shape,
                    dtype=problem.ksp_acc.dtype,
                    device=device,
                    generator=generator_acc,
                )
            if replica:
                xr, _ = ref_system.reconstruct_noise(problem.ksp_ref + nr)
                xa, _ = acc_system.reconstruct_noise(problem.ksp_acc + na)
                for x, mean, m2 in ((xr, mean_ref, sum_ref), (xa, mean_acc, sum_acc)):
                    xd = x.to(torch.complex128)
                    delta = xd - mean
                    mean += delta / index
                    m2 += (delta.conj() * (xd - mean)).real / (sigma_scale ** 2)
            else:
                xr, _ = ref_system.reconstruct_noise(nr)
                xa, _ = acc_system.reconstruct_noise(na)
                sum_ref += xr.abs().double().square()
                sum_acc += xa.abs().double().square()
            if index in checkpoints:
                maps = _reference_maps(problem, sum_ref, sum_acc, index)
                _save_reference_maps(out, f"matched_pmr_N{index}", maps)
                progress_rows.append(
                    {
                        "method": "PMR",
                        "N": index,
                        "cumulative_time_s": _synced_time() - method_start,
                    }
                )
                _append_or_replace_csv(out / "matched_progress.csv", progress_rows)
                print(f"[matched] R={args.R} PMR N={index}", flush=True)

    if "pico" in args.methods:
        method_start = _synced_time()
        generator = torch.Generator(device=device).manual_seed(args.seed + 601)
        diag_ref = torch.zeros(problem.image.shape, dtype=torch.float64, device=device)
        diag_acc = torch.zeros_like(diag_ref)
        for index in range(1, args.N_max + 1):
            phase = 2 * torch.pi * torch.rand(
                problem.image.shape, device=device, generator=generator
            )
            probe = torch.polar(torch.ones_like(phase), phase).type(complex_dtype)
            yr, _, _ = ref_system.covariance_apply(probe)
            ya, _, _ = acc_system.covariance_apply(probe)
            diag_ref += (probe.conj() * yr).real.double()
            diag_acc += (probe.conj() * ya).real.double()
            if index in checkpoints:
                dr = (diag_ref / index).cpu().numpy()
                da = (diag_acc / index).cpu().numpy()
                ratio = np.sqrt(
                    np.maximum(da, 0.0) / np.maximum(dr, 1e-30)
                ) / math.sqrt(problem.R)
                _save_npy(out / f"matched_pico_N{index}_diag_ref.npy", dr)
                _save_npy(out / f"matched_pico_N{index}_diag_acc.npy", da)
                _save_npy(out / f"matched_pico_N{index}_ratio_normalized.npy", ratio)
                ratio_common = ratio * math.sqrt(
                    problem.max_eigen_ref / problem.max_eigen_acc
                )
                _save_npy(
                    out / f"matched_pico_N{index}_ratio_common_units.npy",
                    ratio_common,
                )
                progress_rows.append(
                    {
                        "method": "PICO_random_phase",
                        "N": index,
                        "cumulative_time_s": _synced_time() - method_start,
                    }
                )
                _append_or_replace_csv(out / "matched_progress.csv", progress_rows)
                print(f"[matched] R={args.R} PICO N={index}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["reference", "scaling", "lambda", "matched"]
    )
    parser.add_argument("--R", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--lambda-normalized", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-physical",
        type=float,
        default=None,
        help=(
            "Common PHYSICAL Tikhonov lambda. When given, each arm uses "
            "lambda_normalized = lambda_physical / max_eigen(arm), so both arms "
            "carry the same physical regularization -- the convention used by the "
            "final grid. Without it, --lambda-normalized is applied to both arms, "
            "which gives the reference arm ~2x the accelerated arm's physical "
            "lambda and changes the reported sigma_acc/(sqrt(R) sigma_ref) ratio."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--true-residual-every", type=int, default=10)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))

    parser.add_argument("--N-max", type=int, default=50000)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[10000, 20000, 30000, 50000]
    )
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[25, 50, 100, 200, 400, 800, 1200]
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3, 3e-3]
    )
    parser.add_argument("--n-probes", type=int, default=16)
    parser.add_argument("--methods", nargs="+", choices=["pmr", "pico"], default=["pico"])
    parser.add_argument(
        "--pmr-common-noise",
        action="store_true",
        help="draw one noise realization and use its shot subset for the accelerated arm, so "
             "PMR gets the same common-random-numbers treatment PICO gets from its shared probe.",
    )
    parser.add_argument(
        "--pmr-sigma-scale",
        type=float,
        default=1.0,
        help="multiply the injected PMR noise level by this factor (replica estimator only); "
             "the stored maps are divided by its square, and the normalized ratio map is "
             "independent of the level for a linear reconstruction.",
    )
    parser.add_argument(
        "--pmr-estimator",
        choices=["noise_only", "replica"],
        default="noise_only",
        help="noise_only: accumulate E|R(n)|^2 (exact for a linear R). "
             "replica: Robson-style mean/variance across reconstructions of ksp + n.",
    )
    args = parser.parse_args()
    if args.mode == "scaling" and args.checkpoints == [10000, 20000, 30000, 50000]:
        args.checkpoints = [1000, 2500, 5000]
    if args.mode == "matched" and args.checkpoints == [10000, 20000, 30000, 50000]:
        args.checkpoints = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    return args


def main():
    args = parse_args()
    print(
        f"[noncart-weekend] mode={args.mode} R={args.R} "
        f"job={os.environ.get('SLURM_JOB_ID')} device_cuda={torch.cuda.is_available()}",
        flush=True,
    )
    if args.mode == "reference":
        run_reference(args)
    elif args.mode == "scaling":
        run_scaling(args)
    elif args.mode == "lambda":
        run_lambda(args)
    elif args.mode == "matched":
        run_matched(args)


if __name__ == "__main__":
    main()
