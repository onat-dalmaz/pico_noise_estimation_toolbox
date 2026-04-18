"""End-to-end verification of the three PICO demonstration notebooks.

This script executes ``notebooks/01_cartesian_knee.ipynb``,
``notebooks/02_noncartesian_spiral.ipynb``, and
``notebooks/03_compressed_sensing.ipynb`` in-place using ``nbclient`` and
confirms that

  1. every cell executed without raising (including the § 10 "Verification
     checkpoint" cell, whose ``assert`` statements encode the numeric targets
     derived from Dalmaz et al., *Fast Voxelwise SNR Estimation for Iterative
     MRI Reconstructions*);
  2. each notebook produced its expected figure artifacts under
     ``docs/figures/`` with non-zero size.

Usage::

    python scripts/verify_notebooks.py              # run all three
    python scripts/verify_notebooks.py --only 1     # just notebook 01
    python scripts/verify_notebooks.py --timeout 2400

The script also exposes a ``pytest`` entry point so that ``pytest scripts/``
runs the same verification.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
FIG_DIR = REPO_ROOT / "docs" / "figures"


@dataclass(frozen=True)
class NotebookTarget:
    key: str                    # e.g. "01"
    notebook_path: Path
    expected_figures: tuple[str, ...]


TARGETS: tuple[NotebookTarget, ...] = (
    NotebookTarget(
        key="01",
        notebook_path=NOTEBOOK_DIR / "01_cartesian_knee.ipynb",
        expected_figures=(
            "fig2_cartesian_knee.png",
            "fig2_cartesian_knee_convergence.png",
        ),
    ),
    NotebookTarget(
        key="02",
        notebook_path=NOTEBOOK_DIR / "02_noncartesian_spiral.ipynb",
        expected_figures=(
            "fig3_noncartesian_spiral.png",
            "fig4_noncartesian_spiral_convergence.png",
        ),
    ),
    NotebookTarget(
        key="03",
        notebook_path=NOTEBOOK_DIR / "03_compressed_sensing.ipynb",
        expected_figures=(
            "fig6_compressed_sensing.png",
            "fig6_compressed_sensing_convergence.png",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pretty log helpers
# ---------------------------------------------------------------------------


def _status(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _pass(msg: str) -> None:
    _status("PASS", msg)


def _fail(msg: str) -> None:
    _status("FAIL", msg)


def _info(msg: str) -> None:
    _status("INFO", msg)


# ---------------------------------------------------------------------------
# Notebook runner
# ---------------------------------------------------------------------------


def _execute_notebook(nb_path: Path, timeout: int, kernel_name: str) -> float:
    """Execute a notebook and save it in place. Returns wall-clock seconds."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(REPO_ROOT)}},
        allow_errors=False,
    )
    t0 = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - t0
    nbformat.write(nb, nb_path)
    return elapsed


def _verify_figures(target: NotebookTarget) -> list[str]:
    problems = []
    for name in target.expected_figures:
        path = FIG_DIR / name
        if not path.exists():
            problems.append(f"missing figure: {path}")
        elif path.stat().st_size == 0:
            problems.append(f"empty figure file: {path}")
    return problems


def verify_notebook(target: NotebookTarget, timeout: int, kernel_name: str) -> bool:
    _info(f"executing {target.notebook_path.relative_to(REPO_ROOT)} ...")
    if not target.notebook_path.exists():
        _fail(f"notebook not found: {target.notebook_path}")
        return False
    try:
        elapsed = _execute_notebook(target.notebook_path, timeout, kernel_name)
    except Exception as exc:  # nbclient raises CellExecutionError, etc.
        _fail(f"{target.notebook_path.name} raised during execution: {exc}")
        return False
    _pass(f"{target.notebook_path.name} executed in {elapsed:.1f} s")

    problems = _verify_figures(target)
    if problems:
        for p in problems:
            _fail(p)
        return False
    _pass(f"{target.notebook_path.name} saved expected figures under {FIG_DIR.relative_to(REPO_ROOT)}")
    return True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _select_targets(only: Iterable[str] | None) -> list[NotebookTarget]:
    if not only:
        return list(TARGETS)
    wanted = {o.strip() for o in only}
    chosen = [t for t in TARGETS if t.key in wanted]
    if not chosen:
        raise SystemExit(f"No targets matched --only={sorted(wanted)}; known keys: {[t.key for t in TARGETS]}")
    return chosen


def run(only: list[str] | None = None, timeout: int = 3600, kernel_name: str = "python3") -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    targets = _select_targets(only)
    all_ok = True
    for t in targets:
        ok = verify_notebook(t, timeout=timeout, kernel_name=kernel_name)
        all_ok = all_ok and ok
    if all_ok:
        _pass("all notebooks verified.")
        return 0
    _fail("one or more notebooks failed verification.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", action="append", default=None,
        help="Notebook key to run (e.g. 01, 02, 03). Pass multiple times for subset."
    )
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-cell execution timeout in seconds (default: 3600).")
    parser.add_argument("--kernel-name", default="python3",
                        help="Jupyter kernel name (default: python3).")
    args = parser.parse_args(argv)
    return run(only=args.only, timeout=args.timeout, kernel_name=args.kernel_name)


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------


def _pytest_params():
    import pytest  # noqa: F401
    return TARGETS


try:  # pragma: no cover - only runs under pytest
    import pytest

    @pytest.mark.parametrize("target", TARGETS, ids=[t.key for t in TARGETS])
    def test_notebook(target: NotebookTarget) -> None:
        """Execute each notebook end-to-end. Runs under ``pytest scripts/``."""
        ok = verify_notebook(target, timeout=3600, kernel_name="python3")
        assert ok, f"Notebook verification failed for {target.notebook_path.name}"

except ImportError:  # pytest not installed; skip parametrization
    pass


if __name__ == "__main__":
    sys.exit(main())
