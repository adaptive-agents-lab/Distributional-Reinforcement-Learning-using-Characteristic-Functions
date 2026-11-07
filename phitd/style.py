"""Shared typography and accessible colors for paper figures."""

from pathlib import Path

COLORS = {
    "pqn": "#000000",
    "categorical": "#0072B2",
    "quantile": "#E69F00",
    "dirac": "#CC79A7",
    "gaussian": "#009E73",
    "cauchy": "#D55E00",
    "gamma": "#A68A00",
    "laplace": "#56B4E9",
}
MARKERS = dict(zip(COLORS, (None, "o", "s", "D", "^", "v", "+", "x")))


def algo_color(name):
    aliases = {"phitd_fm": "dirac", "phitd_fcm": "categorical", "phitd_fqm": "quantile"}
    baseline = {
        "ctd": "#D55E00",
        "c51": "#D55E00",
        "qtd": "#009E73",
        "qr_dqn": "#009E73",
    }
    return baseline[name] if name in baseline else COLORS[aliases.get(name, name)]


def configure_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes_panel(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color="#e1e1e1", alpha=0.3)
    ax.set_axisbelow(True)


def save_pdf(fig, path):
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.08)
