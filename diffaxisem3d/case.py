import hashlib
import json
import math
from pathlib import Path
from string import Template

import netCDF4
import numpy as np


CASE = Path(__file__).resolve().parents[1] / "examples" / "salt"


def load_case():
    case = json.loads((CASE / "case.json").read_text())
    path = CASE / "model_fields.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["model_sha256"]:
        raise ValueError("Salt model checksum mismatch")
    with np.load(path) as archive:
        fields = dict(archive)
    return case, fields


def write_material(path, fields, vp):
    axes = ("x", "y", "depth")
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as nc:
        for axis in axes:
            nc.createDimension(axis, len(fields[axis]))
            nc.createVariable(axis, "f8", (axis,))[:] = fields[axis]
        for name, values in [("VP", vp), ("VS", fields["vs"]), ("RHO", fields["rho"])]:
            nc.createVariable(name, "f8", axes)[:] = values


def prepare_shot(path, source, receivers, material):
    path.mkdir()
    for file in (CASE / "input").iterdir():
        (path / file.name).symlink_to(file)
    (path / "model.nc").symlink_to(material)
    x, y, depth = source["xyz_depth_m"]
    text = Template((CASE / "source.yaml.in").read_text()).substitute(
        shot=source["id"],
        radius=math.hypot(x, y),
        azimuth=math.atan2(y, x) % (2 * math.pi),
        depth=depth,
    )
    (path / "inparam.source.yaml").write_text(text)
    positions = ((x, y) for y in receivers["y_m"] for x in receivers["x_m"])
    (path / "stations.txt").write_text(
        "".join(
            f'R{i:03d} XX {x:.17g} {y:.17g} {receivers["depth_m"]:.17g}\n'
            for i, (x, y) in enumerate(positions)
        )
    )
