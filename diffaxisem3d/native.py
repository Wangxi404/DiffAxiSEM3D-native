"""AxiSEM3D receiver sampling, native adjoint and Vp chain rule."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock

import numpy as np

from .case import prepare_shot, write_material
from .material import NativeMaterialMap, read_material


def read_receivers(prefix, ranks):
    parts = []
    for rank in range(ranks):
        stem = f"{prefix}.rank{rank}"
        meta = json.loads(Path(stem + ".stations.json").read_text())
        if (meta["nt"], meta["dt"], meta["t0"], meta["components"]) != (
            1051,
            0.01,
            -2.5,
            ["S", "P", "Z"],
        ):
            raise ValueError("Unexpected native time grid or displacement components")
        raw = np.fromfile(stem + ".data.bin", dtype="<f8").reshape(meta["shape"])
        parts.append((meta["keys"], raw))
    keys = sorted(key for names, _ in parts for key in names)
    if keys != [f"XX.R{i:03d}" for i in range(169)]:
        raise ValueError("Duplicate or missing receivers in MPI output")
    indices = {key: i for i, key in enumerate(keys)}
    data = np.empty((1051, len(keys), 3))
    partition = []
    for names, raw in parts:
        ids = [indices[key] for key in names]
        data[:, ids, :] = raw
        partition.append(ids)
    return data, partition


class NativeSolver:
    def __init__(self, exe, case, fields, objective, work, workers, ranks):
        self.exe = Path(exe).resolve()
        self.work = Path(work).resolve()
        self.fields = fields
        self.sources = case["acquisition"]["sources"]
        self.receivers = case["acquisition"]["receiver_grid"]
        self.objective = objective
        self.workers = workers
        self.ranks = ranks
        self.observed = {}
        self.maps = {}
        self.lock = Lock()

    def launch(self, mode, path):
        env = dict(
            os.environ, AXISEM_DIFF_MODE=mode, AXISEM_DIFF_PREFIX=str(path / "native")
        )
        cmd = [
            "mpirun",
            "--bind-to",
            "none",
            "-np",
            str(self.ranks),
            str(self.exe),
            "--input",
            str(path / "input"),
            "--output",
            str(path / mode),
        ]
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=path,
        )
        if result.returncode:
            raise RuntimeError(
                f"AxiSEM3D {mode} failed ({result.returncode})\n"
                + "\n".join(result.stdout.splitlines()[-50:])
            )

    def vp_gradient(self, model, prefix, rank):
        records = read_material(f"{prefix}.rank{rank}.material.bin")
        signature = sha256()
        for part in records:
            signature.update(
                np.array([part["tag"], part["nr"], part["p"]], dtype="<i8").tobytes()
            )
            signature.update(part["point_nr"].tobytes())
            signature.update(part["sz"].tobytes())
        key = signature.digest()
        with self.lock:
            if key not in self.maps:
                axes = [self.fields[k] for k in ("x", "y", "depth")]
                self.maps[key] = NativeMaterialMap(records, axes)
            mapping = self.maps[key]
        vp_native = np.concatenate([part["vp"].ravel() for part in records])
        lam_native = np.concatenate([part["lam"].ravel() for part in records])
        if np.linalg.norm(
            mapping.T @ model.ravel() - vp_native
        ) > 3e-12 * np.linalg.norm(vp_native) or np.linalg.norm(
            mapping.coefficients(model) - lam_native
        ) > 6e-12 * np.linalg.norm(
            lam_native
        ):
            raise ValueError(
                "Vp controls disagree with the initialized native material"
            )
        return mapping.vjp(model, mapping.read_lambda(prefix, rank))

    def shot(self, source, model, material, observation, gradient):
        shot = source["id"]
        with tempfile.TemporaryDirectory(prefix=shot + "-", dir=self.work) as tmp:
            path = Path(tmp)
            prepare_shot(path / "input", source, self.receivers, material)
            self.launch("record" if gradient else "forward", path)
            prefix = path / "native"
            predicted, partition = read_receivers(prefix, self.ranks)
            if observation:
                scale = float(np.sqrt(np.mean(predicted**2)))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid observed RMS for {shot}")
                self.observed[shot] = (predicted, scale)
                return None
            observed, scale = self.observed[shot]
            loss, cotangent = self.objective.value_gradient(
                predicted, observed, 0.01, scale, partition, shot, gradient
            )
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {shot}")
            if not gradient:
                return loss, None
            if not np.isfinite(cotangent).all():
                raise FloatingPointError(f"Nonfinite receiver derivative for {shot}")
            for rank, stations in enumerate(partition):
                cotangent[:, stations, :].astype("<f8").tofile(
                    f"{prefix}.rank{rank}.cot.bin"
                )
            self.launch("adjoint", path)
            g = sum(
                (self.vp_gradient(model, prefix, rank) for rank in range(self.ranks)),
                np.zeros_like(model),
            )
            if not np.isfinite(g).all():
                raise FloatingPointError(f"Nonfinite model gradient for {shot}")
            return loss, g

    def evaluate(self, model, *, gradient=False, observation=False):
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            material = Path(tmp) / "model.nc"
            write_material(material, self.fields, model)
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [
                    pool.submit(
                        self.shot, source, model, material, observation, gradient
                    )
                    for source in self.sources
                ]
                results = [f.result() for f in futures]
        if observation:
            return
        loss = sum(result[0] for result in results) / len(results)
        g = (
            sum((result[1] for result in results), np.zeros_like(model)) / len(results)
            if gradient
            else None
        )
        return loss, g
