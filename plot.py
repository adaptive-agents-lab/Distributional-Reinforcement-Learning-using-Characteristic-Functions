"""Generate the MinAtar and Craftax figures from local seed curves."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from phitd.style import COLORS, MARKERS, configure_matplotlib, save_pdf

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
METHODS = list(COLORS)
ENVS = [
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
]


def smooth(values, window):
    values = np.asarray(values, dtype=float).copy()
    window = min(window | 1, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values
    bad = ~np.isfinite(values)
    if bad.any() and (~bad).any():
        values[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), values[~bad])
    return np.convolve(
        np.pad(values, window // 2, mode="edge"), np.ones(window) / window, mode="valid"
    )


def aggregate(curves, window):
    steps = np.concatenate([np.asarray(c["steps"]) for c in curves])
    if len(steps) < 2 or steps.max() <= steps.min():
        raise ValueError(
            "A curve needs at least two distinct steps; train longer before plotting."
        )
    grid = np.linspace(steps.min(), steps.max(), 800)
    values = np.stack(
        [
            np.interp(grid, c["steps"], c["returns"], left=np.nan, right=np.nan)
            for c in curves
        ]
    )
    count = np.isfinite(values).sum(axis=0)
    mean = smooth(np.nanmean(values, axis=0), window)
    spread = 1.96 * np.nanstd(values, axis=0) / np.sqrt(np.maximum(count, 1))
    spread[count < 2] = np.nan
    spread = smooth(spread, window)
    spread[count < 2] = np.nan
    return grid, mean, spread


def load_curves(data_dir, experiment):
    snapshot = data_dir / f"{experiment}.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())["curves"]
    files = sorted((data_dir / experiment).glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No {experiment} curves under {data_dir}. Run train.py {experiment} first."
        )
    return [curve for path in files for curve in json.loads(path.read_text())["curves"]]


def plot(experiment, curves, output):
    minatar = experiment == "minatar"
    envs = ENVS if minatar else ["Craftax-Symbolic-v1"]
    # Preserve the paper's family ordering and dimensions.
    order = (
        [
            "pqn",
            "gaussian",
            "gamma",
            "laplace",
            "categorical",
            "quantile",
            "dirac",
            "cauchy",
        ]
        if minatar
        else METHODS[1:] + ["pqn"]
    )
    fig, axes = plt.subplots(
        1, len(envs), figsize=(23, 5.2) if minatar else (14, 8), squeeze=False
    )
    for panel, (ax, env) in enumerate(zip(axes[0], envs)):
        lower, upper = np.inf, -np.inf
        for method in order:
            group = [
                c
                for c in curves
                if c["method"] == method and c["env"] == env and len(c["steps"]) >= 2
            ]
            if not group:
                raise ValueError(
                    f"Missing seed curves for {experiment}/{env}/{method}."
                )
            x, y, ci = aggregate(group, 41 if minatar else 51)
            label = (
                ("PQN" if minatar else "PQN-RNN")
                if method == "pqn"
                else rf"$\varphi\text{{TD-{method.title()}}}$"
            )
            ax.plot(x, y, color=COLORS[method], linewidth=2, label=label)
            ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.2)
            if MARKERS[method]:
                index = METHODS.index(method)
                mx = x[0] + (x[-1] - x[0]) * (
                    np.array([0.13, 0.31, 0.49, 0.67, 0.85]) + (index - 4) * 0.012
                )
                ax.plot(
                    mx,
                    np.interp(mx, x, y),
                    linestyle="none",
                    marker=MARKERS[method],
                    markersize=10.5 if minatar else 8.5,
                    markerfacecolor="white",
                    markeredgecolor=COLORS[method],
                    markeredgewidth=1.5,
                )
            band = np.concatenate([y - ci, y + ci, y])
            lower = min(lower, np.nanmin(band))
            upper = max(upper, np.nanmax(band))
        pad = max(upper - lower, 1e-6) * 0.03
        ax.set_ylim(0, upper + pad)
        ax.set_xlabel("Environment Steps", fontsize=20 if minatar else 26)
        if panel == 0:
            ax.set_ylabel("Episode Return", fontsize=20 if minatar else 26)
        ax.tick_params(labelsize=16 if minatar else 20)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if minatar:
            ax.set_title(env.removesuffix("-MinAtar"), fontsize=20)
        if panel == 0:
            from matplotlib.lines import Line2D

            handles, labels = ax.get_legend_handles_labels()
            handles = [
                Line2D(
                    [],
                    [],
                    color=COLORS[method],
                    linewidth=2,
                    marker=MARKERS[method],
                    markersize=10.5 if minatar else 8.5,
                    markerfacecolor="white",
                )
                for method in order
            ]
            ax.legend(handles, labels, fontsize=14 if minatar else 16, loc="best")
    fig.tight_layout()
    save_pdf(fig, output / f"{experiment}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--experiment", choices=("minatar", "craftax"))
    args = parser.parse_args()
    configure_matplotlib()
    for experiment in [args.experiment] if args.experiment else ["minatar", "craftax"]:
        plot(experiment, load_curves(args.data_dir, experiment), args.output)
        print(f"Saved {args.output / (experiment + '.pdf')}")


if __name__ == "__main__":
    main()
