"""Plot salt-model recovery and the full-shot loss history."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODEL_NAMES = ("true", "initial", "inverted")


def cell_bounds(coords):
    return coords[0] - (coords[1] - coords[0]) / 2, coords[-1] + (coords[-1] - coords[-2]) / 2


def shared_vp_norm(data):
    mask = data["display_mask"]
    if mask.dtype != bool or not np.any(mask):
        raise ValueError("Expected a nonempty boolean simulation-domain mask.")
    limits = []
    for name in MODEL_NAMES:
        vp = data[name][mask] / 1000
        if not np.isfinite(vp).all():
            raise ValueError(f"Nonfinite {name} Vp inside the simulated domain.")
        limits.append((vp.min(), vp.max()))
    return matplotlib.colors.Normalize(min(low for low, _ in limits), max(high for _, high in limits))


def model_figure(data, acquisition):
    x, y, z = (data[key] / 1000 for key in ("x", "y", "depth"))
    mask = data["display_mask"]
    size, gap, left = 40.5, 8, 20
    xy_height = size * np.ptp(y) / np.ptp(x)
    xz_height = size * np.ptp(z) / np.ptp(x)
    yz_height = size * np.ptp(z) / np.ptp(y)
    rows = [(36 + xz_height + yz_height, xy_height), (25 + yz_height, xz_height), (14, yz_height)]
    width, height = 210, rows[0][0] + xy_height + 13
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))

    def panel(left, bottom, w, h):
        return fig.add_axes([left / width, bottom / height, w / width, h / height])

    cmap = "Blues"
    norm = shared_vp_norm(data)
    planes = ("X–Y plane", "X–Z plane", "Y–Z plane")
    vertical = (y, z, z)
    horizontal = (x, x, y)
    sections = (
        (slice(None), slice(None), len(z) // 2),
        (slice(None), len(y) // 2, slice(None)),
        (len(x) // 2, slice(None), slice(None)),
    )

    for row, (bottom, h) in enumerate(rows):
        u, v = horizontal[row], vertical[row]
        for col, name in enumerate(MODEL_NAMES):
            ax = panel(left + col * (size + gap), bottom, size, h)
            cut = sections[row]
            values = np.ma.array(data[name][cut].T / 1000, mask=~mask[cut].T)
            ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest", origin="lower",
                      extent=(*cell_bounds(u), *cell_bounds(v)), aspect="equal")
            ax.set(xlim=(u[0], u[-1]), ylim=(v[0], v[-1]) if row == 0 else (v[-1], v[0]))
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.xaxis.set_visible(row == 2 or (row == 1 and col == 0))
            ax.yaxis.set_visible(col == 0)
            if row == 0:
                ax.set_title(name.capitalize(), loc="left", pad=7)
            if row == 1 and col == 0:
                ax.set_xlabel("x (km)", labelpad=1)
            if col == 0:
                centre = (bottom + h / 2) / height
                fig.text(10 / width, centre, "y (km)" if row == 0 else "Depth (km)",
                         ha="center", va="center", rotation=90, rotation_mode="anchor")
                fig.text(2.5 / width, centre, planes[row], fontsize=7.5,
                         ha="center", va="center", rotation=90, rotation_mode="anchor")
            panel_letter(ax, ("abc", "efg", "hij")[row][col])

    right = left + 3 * (size + gap)
    ax = panel(right, rows[0][0], size, xy_height)
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xticks=[], yticks=[], aspect="equal")
    ax.set_title("Acquisition (X–Y)", loc="left", pad=7)
    grid = acquisition["receiver_grid"]
    rx, ry = np.meshgrid(np.asarray(grid["x_m"]) / 1000, np.asarray(grid["y_m"]) / 1000)
    sources = np.array([source["xyz_depth_m"] for source in acquisition["sources"]]) / 1000
    receivers = dict(marker="^", color="black", markeredgewidth=0, linestyle="none")
    shots = dict(marker="*", color="#e31a1c", markeredgecolor="#7a0000", markeredgewidth=.15,
                 linestyle="none")
    ax.plot(rx.ravel(), ry.ravel(), markersize=1.8, **receivers)
    ax.plot(sources[:, 0], sources[:, 1], markersize=5, **shots)
    panel_letter(ax, "d")
    fig.legend([Line2D([], [], markersize=3.5, **receivers), Line2D([], [], markersize=7, **shots)],
               ["Receivers", "Sources"], ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=((right + size / 2) / width, rows[1][0] / height), borderaxespad=0,
               handlelength=1, handletextpad=.35, columnspacing=1, fontsize=7.5)
    color_axis = panel(right + (size - 36) / 2, rows[1][0] - 6, 36, 2.5)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                      orientation="horizontal",
                      ticks=matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    cb.set_label("Vp (km/s)", fontsize=7)
    fig.text((left + 1.5 * size + gap) / width, 3.5 / height, "y (km)", ha="center", va="bottom")
    return fig


def panel_letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-10, 7),
                textcoords="offset points", ha="right", va="baseline", weight="bold", fontsize=9)


def loss_figure(loss):
    fig, ax = plt.subplots(figsize=(6.5, 1.65))
    fig.subplots_adjust(left=.14, bottom=.29, right=.96, top=.92)
    ax.plot(np.arange(len(loss)), loss, color="#2166ac", linewidth=1.15,
            marker="o" if len(loss) < 3 else None, markersize=3)
    ax.set(xlabel="Accepted update", ylabel="Loss")
    ax.set_xlim(0, max(len(loss) - 1, 1))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def render(models_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_file = Path(__file__).resolve().parents[1] / "examples/salt/case.json"
    acquisition = json.loads(case_file.read_text())["acquisition"]
    with np.load(models_path, allow_pickle=False) as data:
        updates = int(data["accepted_update"].item())
        loss = data["loss"]
        if loss.shape != (updates + 1,) or not np.all(np.isfinite(loss)):
            raise ValueError("Expected one finite loss for the initial model and each accepted update.")
        x_mid, y_mid, z_mid = (data[key][len(data[key]) // 2] / 1000 for key in ("x", "y", "depth"))
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                             "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": .45,
                             "xtick.major.width": .4, "ytick.major.width": .4, "pdf.fonttype": 42}):
            model_plot = model_figure(data, acquisition)
            low, high = model_plot.axes[0].images[0].get_clim()
            for name, fig in (("models", model_plot), ("loss", loss_figure(loss))):
                credit = {"Description": "Resampled SEG/EAGE salt model, copyright 1997 Society of "
                          "Exploration Geophysicists, https://creativecommons.org/licenses/by/4.0/"}
                fig.savefig(output / f"{name}.png", dpi=300, facecolor="white",
                            metadata=credit if name == "models" else None)
                plt.close(fig)

    report = output / "model-recovery.html"
    report.write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffAxiSEM3D-native · Salt-model recovery</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#172c40;max-width:960px;margin:auto;padding:3vw}}
h1{{font-size:1.55rem;margin-bottom:0}}p{{margin:.3em 0 1em}}img{{width:100%;height:auto}}
figure{{margin:1.3em 0}}figcaption{{font-size:.9rem;color:#455567}}h2{{font-size:1.1rem}}</style>
<main><h1>Salt-model recovery</h1><p>DiffAxiSEM3D-native · {updates} accepted updates</p>
<figure><img src="models.png" alt="True, initial and inverted Vp in three central slices, with acquisition geometry">
<figcaption>Central horizontal slice at {z_mid:g} km depth and vertical slices at y = {y_mid:g} km
and x = {x_mid:g} km. Horizontal and vertical distances use the same scale. All panels use the full Vp
range of the three model volumes inside the simulated cylinder ({low:g}–{high:g} km/s).
The resampled SEG/EAGE salt model is © 1997 Society of Exploration Geophysicists
(<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>).</figcaption></figure>
<h2>Loss</h2><img src="loss.png" alt="Mean full-shot loss from the initial model through the last accepted update">
</main></html>
""", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    args = parser.parse_args()
    print(render(args.models, args.models.parent))
