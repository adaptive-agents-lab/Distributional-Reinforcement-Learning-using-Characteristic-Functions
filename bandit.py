"""Train the contextual-bandit benchmark and save distribution and Parseval figures."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from phitd import bandit as model
from phitd.style import configure_matplotlib, save_pdf

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent


def train(cfg):
    ri, rd, _, rng = jax.random.split(jax.random.PRNGKey(cfg.seed), 4)
    dataset = model.build_dataset(jax.random.fold_in(rd, cfg.dataset_seed_offset), cfg)
    ctd, qtd, dirac, categorical, quantile, *nets = model.create_train_states(ri, cfg)
    states = (ctd, qtd, dirac, categorical, quantile)
    evaluate, frequencies, returns = model._make_eval(cfg, *nets)
    anchor = jax.random.normal(
        model._panel_anchor_prng_key(), (cfg.state_dim,), jnp.float32
    )
    update = model.make_train_chunk(cfg, *nets, dataset)
    history = {"step": [], "cramer_l2_dirac": [], "cf_l2_w_dirac": []}
    for step in range(cfg.parseval_every, cfg.total_steps + 1, cfg.parseval_every):
        *states, rng, losses = update(*states, rng)
        if not all(np.isfinite(np.asarray(loss)).all() for loss in losses):
            raise RuntimeError(f"Non-finite loss at step {step}, seed {cfg.seed}")
        metrics = evaluate(*(s.params for s in states), anchor)
        history["step"].append(step)
        for key in ("cramer_l2_dirac", "cf_l2_w_dirac"):
            history[key].append(float(jnp.mean(metrics[key])))
    payload = {
        "seed": cfg.seed,
        "omega_max": cfg.omega_max,
        "t_eval": np.asarray(frequencies).tolist(),
        "x_eval": np.asarray(returns).tolist(),
        "panel": {
            f"panel_{key}": np.asarray(value).real.tolist()
            for key, value in metrics.items()
            if key.startswith(("phi_", "pdf_", "F_"))
        },
    }
    return payload, history


def plot(records, output):
    payloads = [r["payload"] for r in records]
    frequencies = np.asarray(payloads[0]["t_eval"])
    returns = np.asarray(payloads[0]["x_eval"])
    width = payloads[0]["omega_max"]
    mean, ci = {}, {}
    for key in payloads[0]["panel"]:
        values = np.stack([p["panel"][key] for p in payloads])
        mean[key] = np.mean(values, axis=0)
        ci[key] = model._mean_ci95_halfwidth(values)
    output.mkdir(parents=True, exist_ok=True)
    model._plot_panels(
        mean, ci, frequencies, returns, width, str(output / "bandit.pdf")
    )
    history = model._merge_parseval_histories([r["history"] for r in records])
    steps = np.asarray(history["step"])
    fig, ax = plt.subplots(figsize=(6.1, 3.5))
    for key, label, color, dash in [
        ("cramer_l2_dirac", r"Cramér $L_2^2$ (CDF)", "#0072B2", "-"),
        ("cf_l2_w_dirac", r"CF $L_2^2/\omega^2$", "#D55E00", "--"),
    ]:
        values = np.atleast_2d(history[key])
        y = np.mean(values, axis=0)
        half = model._mean_ci95_halfwidth(values)
        ax.plot(steps, y, label=label, color=color, linestyle=dash)
        ax.fill_between(
            steps, np.maximum(y - half, 0), y + half, color=color, alpha=0.2
        )
    ax.set(xlabel="Training Step", ylabel="Loss", yscale="log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_pdf(fig, output / "parseval.pdf")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "bandit.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short training/evaluation check, not paper results.",
    )
    parser.add_argument(
        "--replot", action="store_true", help="Plot saved local runs without training."
    )
    args = parser.parse_args()
    cfg = model.load_da_config_yaml(args.config, {})
    if args.smoke:
        cfg = replace(
            cfg,
            num_seeds=2,
            dataset_size=128,
            batch_size=8,
            total_steps=4,
            parseval_every=2,
            hidden_dim=16,
            trunk_layers=1,
            num_tau=4,
            num_atoms=4,
            num_dirac_components=4,
            num_omega=4,
            eval_num_t=31,
            eval_num_x=51,
            gp_num_t=101,
        )
    if (
        cfg.num_seeds < 1
        or cfg.total_steps < 1
        or cfg.parseval_every < 1
        or cfg.total_steps % cfg.parseval_every
    ):
        parser.error(
            "Use positive seeds/steps and total_steps divisible by parseval_every."
        )
    results = ROOT / "results" / ("smoke" if args.smoke else "") / "bandit"
    output = ROOT / "figures" / ("smoke" if args.smoke else "")
    results.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in range(cfg.seed, cfg.seed + cfg.num_seeds):
        path = results / f"seed{seed}.json"
        settings = replace(cfg, seed=seed)
        if args.replot:
            record = json.loads(path.read_text())
            if record["config"] != asdict(settings):
                raise ValueError(f"Settings differ from saved run {path}")
        else:
            print(f"Bandit: seed {seed}", flush=True)
            payload, history = train(settings)
            record = {
                "config": asdict(settings),
                "payload": payload,
                "history": history,
            }
            path.write_text(json.dumps(record) + "\n")
        records.append(record)
    configure_matplotlib()
    plot(records, output)
    print(f"Saved {output / 'bandit.pdf'} and {output / 'parseval.pdf'}")


if __name__ == "__main__":
    main()
