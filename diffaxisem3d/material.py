"""Map Cartesian Vp controls to native elastic coefficients and back."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def read_material(path):
    records = []
    with open(path, "rb") as f:
        while True:
            header = np.fromfile(f, dtype="<i8", count=3)
            if not len(header):
                break
            if len(header) != 3:
                raise ValueError("truncated material header")
            tag, nr, p = map(int, header)
            point_nr = np.fromfile(f, dtype="<i8", count=p)
            sz = np.fromfile(f, dtype="<f8", count=p * 2).reshape(p, 2)
            vals = np.fromfile(f, dtype="<f8", count=nr * p * 4).reshape(nr, p, 4)
            if (point_nr <= 0).any() or max(point_nr) != nr:
                raise ValueError("invalid point Nr")
            records.append(
                dict(
                    tag=tag,
                    nr=nr,
                    p=p,
                    point_nr=point_nr,
                    sz=sz,
                    vp=vals[:, :, 0],
                    rho=vals[:, :, 1],
                    vs=vals[:, :, 2],
                    lam=vals[:, :, 3],
                )
            )
    return records


def trilinear_matrix(points, axes):
    """Trilinear weights from Cartesian control nodes to native sampling points."""
    points = np.asarray(points)
    n = len(points)
    inds = []
    weights = []
    for j, axis in enumerate(axes):
        axis = np.asarray(axis)
        if (points[:, j] < axis[0] - 1e-6).any() or (
            points[:, j] > axis[-1] + 1e-6
        ).any():
            raise ValueError("material points outside supplied model")
        loc = np.clip(
            np.searchsorted(axis, points[:, j], side="right") - 1, 0, len(axis) - 2
        )
        frac = (points[:, j] - axis[loc]) / (axis[loc + 1] - axis[loc])
        inds.append(loc)
        weights.append(frac)
    rows = []
    cols = []
    data = []
    shape = tuple(map(len, axes))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                rows.append(np.arange(n))
                cols.append(
                    np.ravel_multi_index((inds[0] + a, inds[1] + b, inds[2] + c), shape)
                )
                data.append(
                    (weights[0] if a else 1 - weights[0])
                    * (weights[1] if b else 1 - weights[1])
                    * (weights[2] if c else 1 - weights[2])
                )
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(np.prod(shape))),
    ).tocsr()


class NativeMaterialMap:
    def __init__(self, records, axes, reference_radius=6371000.0):
        self.shape = tuple(map(len, axes))
        self.offsets = {}
        mats = []
        offset = 0
        for r in records:
            nr, p = r["nr"], r["p"]
            q = []
            starts = []
            start = 0
            for ip in range(p):
                np_ = int(r["point_nr"][ip])
                phi = 2 * np.pi * np.arange(np_) / np_
                s, z = r["sz"][ip]
                starts.append(start)
                start += np_
                q.append(
                    np.column_stack(
                        (
                            s * np.cos(phi),
                            s * np.sin(phi),
                            np.full(np_, reference_radius - z),
                        )
                    )
                )
            P = trilinear_matrix(np.concatenate(q), axes)
            rr = []
            cc = []
            vv = []
            for ip in range(p):
                np_ = int(r["point_nr"][ip])
                pos = np.arange(nr) * np_ / nr
                i0 = np.floor(pos).astype(int)
                f = pos - i0
                rr.extend((np.arange(nr) * p + ip, np.arange(nr) * p + ip))
                cc.extend((starts[ip] + i0, starts[ip] + (i0 + 1) % np_))
                vv.extend((1 - f, f))
            R = sparse.coo_matrix(
                (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                shape=(nr * p, start),
            ).tocsr()
            mats.append(R @ P)
            self.offsets[r["tag"]] = (offset, offset + nr * p)
            offset += nr * p
        self.T = sparse.vstack(mats, format="csr")
        self.rho = np.concatenate([r["rho"].ravel() for r in records])
        self.vs = np.concatenate([r["vs"].ravel() for r in records])

    def coefficients(self, model):
        vp = self.T @ np.asarray(model).reshape(-1)
        return self.rho * (vp * vp - 2 * self.vs * self.vs)

    def vjp(self, model, bar_lambda):
        vp = self.T @ np.asarray(model).reshape(-1)
        return np.asarray(
            self.T.T @ (2 * self.rho * vp * np.asarray(bar_lambda).reshape(-1))
        ).reshape(self.shape)

    def read_lambda(self, prefix, rank):
        raw = np.fromfile(f"{prefix}.rank{rank}.lambda.bin", dtype="<f8")
        bar = np.zeros(self.T.shape[0])
        found = []
        for line in Path(f"{prefix}.rank{rank}.lambda.jsonl").read_text().splitlines():
            r = json.loads(line)
            tag = r["qtag"]
            lo, hi = self.offsets[tag]
            offset = r["offset_f64"]
            bar[lo:hi] = raw[offset : offset + hi - lo]
            found.append(tag)
        if sorted(found) != sorted(self.offsets):
            raise ValueError("gradient element IDs differ from material partition")
        return bar
