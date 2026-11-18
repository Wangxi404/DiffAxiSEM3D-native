"""Run the SEG/EAGE salt example with native forward and adjoint propagation."""
import argparse
from hashlib import sha256
import fcntl
import importlib
import os
from pathlib import Path
import shutil
import tempfile

if __name__ == "__main__":
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[variable] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from .case import CASE, load_case, simulation_mask
from .native import NativeSolver
from .objective import SoftDTW, TensorMisfit
from .plotting import render


def lbfgs_direction(gradient, steps, differences):
    q = gradient.copy()
    alphas, rhos = [], []
    for s, y in zip(reversed(steps), reversed(differences)):
        rho = 1.0 / np.sum(s * y)
        alpha = rho * np.sum(s * q)
        alphas.append(alpha)
        rhos.append(rho)
        q -= alpha * y
    scale = (
        np.sum(steps[-1] * differences[-1]) / np.sum(differences[-1] ** 2)
        if steps
        else 1.0
    )
    r = gaussian_filter(q, 0.35, mode="nearest") * scale
    for s, y, alpha, rho in zip(steps, differences, reversed(alphas), reversed(rhos)):
        r += s * (alpha - rho * np.sum(y * r))
    p = -r
    if not np.isfinite(p).all() or np.sum(gradient * p) >= -1e-14 * np.linalg.norm(
        gradient
    ) * max(np.linalg.norm(p), 1e-300):
        steps.clear()
        differences.clear()
        p = -gaussian_filter(gradient, 0.35, mode="nearest")
        if np.sum(gradient * p) >= 0:
            p = -gradient.copy()
    maximum = np.max(np.abs(p))
    if not np.isfinite(maximum) or maximum == 0:
        raise RuntimeError("No finite descent direction")
    return p / maximum


def inversion(solver, initial, iterations, checkpoint, identity):
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            model, previous, previous_grad = (
                saved[k] for k in ("vp", "previous", "gradient")
            )
            steps, diffs, losses = (
                list(saved[k]) for k in ("steps", "differences", "loss")
            )
            scale = float(saved["scale"])
    else:
        model, previous, previous_grad = initial.copy(), None, None
        steps, diffs, losses, scale = [], [], [], 100.0
    accepted = max(0, len(losses) - 1)
    if accepted > iterations:
        raise ValueError(f"Checkpoint already contains {accepted} updates")
    for iteration in range(accepted + 1, iterations + 1):
        loss, g = solver.evaluate(model, gradient=True)
        if not losses:
            losses.append(loss)
        elif abs(loss - losses[-1]) > 1e-9 * max(abs(loss), 1.0):
            raise ValueError("Gradient and line search evaluated different objectives")
        if previous is not None:
            s, y = model - previous, g - previous_grad
            sy = np.sum(s * y)
            if (
                sy > 0
                and sy / max(np.linalg.norm(s) * np.linalg.norm(y), 1e-300) > 1e-8
            ):
                steps, diffs = (steps + [s])[-10:], (diffs + [y])[-10:]
        direction = lbfgs_direction(g, steps, diffs)
        for trial in range(16):
            step = min(scale, 500.0) * 0.5**trial
            candidate = np.clip(model + step * direction, 1450.0, 4700.0)
            slope = np.sum(g * (candidate - model))
            if slope >= 0:
                continue
            value, _ = solver.evaluate(candidate)
            if value <= loss + 1e-4 * slope:
                break
        else:
            raise RuntimeError(f"All line search trials rejected at update {iteration}")
        scale = min(500.0, 1.5 * step)
        losses.append(value)
        previous, previous_grad, model = model, g, candidate
        with tempfile.NamedTemporaryFile(
            dir=checkpoint.parent, suffix=".npz", delete=False
        ) as f:
            np.savez_compressed(
                f,
                vp=model,
                previous=previous,
                gradient=g,
                steps=steps,
                differences=diffs,
                loss=losses,
                scale=scale,
                identity=identity,
            )
            f.flush()
            os.fsync(f.fileno())
        Path(f.name).replace(checkpoint)
        print(f"Update {iteration}/{iterations}   loss {value:.9g}", flush=True)
    return model, np.asarray(losses)


def misfit(name):
    if name is None:
        return SoftDTW(), Path(__file__).with_name("objective.py")
    module, function = name.split(":", 1)
    module = importlib.import_module(module)
    return TensorMisfit(getattr(module, function)), Path(module.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/axisem3d"))
    parser.add_argument("--output", type=Path, default=Path("runs/salt"))
    parser.add_argument(
        "--workers", type=int, default=1, help="Shots to run concurrently (1–9)"
    )
    parser.add_argument(
        "--ranks", type=int, choices=(1, 2), default=2, help="MPI ranks per shot"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=125,
        help="Number of accepted full-shot updates",
    )
    parser.add_argument(
        "--misfit",
        metavar="MODULE:FUNCTION",
        help="PyTorch loss(predicted, observed, dt, observed_rms)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue from this output directory"
    )
    args = parser.parse_args()
    if not args.exe.is_file() or not os.access(args.exe, os.X_OK):
        parser.error(f"Native executable is missing or not executable: {args.exe}")
    if not shutil.which("mpirun"):
        parser.error("mpirun is required")
    if not 1 <= args.workers <= 9 or args.iterations < 1:
        parser.error("Use 1–9 workers and at least one iteration")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another inversion is using this output directory")
        run(args, parser)


def run(args, parser):
    checkpoint = args.output / "checkpoint.npz"
    if args.resume != checkpoint.is_file():
        parser.error(
            "Use --resume for an existing checkpoint, or choose a fresh output directory"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    case, fields = load_case()
    objective, objective_path = misfit(args.misfit)
    fingerprint = sha256()
    for path in (
        args.exe,
        *sorted(p for p in CASE.rglob("*") if p.is_file()),
        objective_path,
        Path(__file__).with_name("soft_dtw.py"),
    ):
        fingerprint.update(path.read_bytes())
    fingerprint.update(f"{args.ranks}:{args.misfit}".encode())
    with tempfile.TemporaryDirectory(prefix="native-", dir=args.output) as work:
        solver = NativeSolver(
            args.exe, case, fields, objective, work, args.workers, args.ranks
        )
        accepted = 0
        if args.resume:
            with np.load(checkpoint) as saved:
                accepted = len(saved["loss"]) - 1
                if str(saved["identity"]) != fingerprint.hexdigest():
                    raise ValueError(
                        "Checkpoint solver, case or misfit differs from this run"
                    )
        if accepted < args.iterations:
            print("Computing nine synthetic shot gathers", flush=True)
            solver.evaluate(fields["true"], observation=True)
        model, loss = inversion(
            solver,
            fields["initial"],
            args.iterations,
            checkpoint,
            fingerprint.hexdigest(),
        )
    path = args.output / "models.npz"
    np.savez_compressed(
        path,
        **{k: fields[k] for k in ("x", "y", "depth", "true", "initial")},
        inverted=model,
        loss=loss,
        accepted_update=len(loss) - 1,
        display_mask=simulation_mask(fields),
    )
    render(path, args.output)
    print(args.output / "model-recovery.html")


if __name__ == "__main__":
    main()
