"""Portable launchers for the MinAtar and Craftax paper experiments."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent


def jobs(experiment, smoke):
    spec = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / f"{experiment}.yaml")
    )
    result = []
    for env in spec["envs"][:1] if smoke else spec["envs"]:
        for method, overrides in spec["methods"].items():
            for seed in spec["seeds"][:1] if smoke else spec["seeds"]:
                cfg = {**spec["common"], **overrides, "ENV_NAME": env, "SEED": seed}
                if smoke:
                    cfg.update(
                        NUM_SEEDS=1,
                        NUM_ENVS=2,
                        NUM_STEPS=8,
                        TOTAL_TIMESTEPS=32,
                        TOTAL_TIMESTEPS_DECAY=32,
                        NUM_MINIBATCHES=1,
                        NUM_EPOCHS=1,
                        LOG_INTERVAL=1,
                        TEST_DURING_TRAINING=False,
                        MLP_HIDDEN_DIM=16,
                        HIDDEN_SIZE=16,
                        NUM_LAYERS=1,
                        MEMORY_WINDOW=0,
                        M_PARTICLES=4,
                        NUM_ATOMS=4,
                        NUM_OMEGA_SAMPLES=4,
                        OPTIMISTIC_RESET_RATIO=2,
                    )
                result.append((method, cfg))
    return result


def run(experiment, method, cfg, output):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import numpy as np

    module = importlib.import_module(
        f"phitd.{'pqn_' if method == 'pqn' else ''}{experiment}"
    )
    seed = cfg["SEED"]
    rngs = jax.random.split(jax.random.PRNGKey(seed), cfg["NUM_SEEDS"])
    seed_indices = jnp.arange(cfg["NUM_SEEDS"], dtype=jnp.int32)
    train = jax.jit(jax.vmap(module.make_train(dict(cfg))))
    trained = jax.block_until_ready(train(rngs, seed_indices))
    metrics = jax.device_get(trained["metrics"])
    # Zero completed episodes can give NaN returns; non-finite losses are failures.
    for key, values in metrics.items():
        if ("loss" in key or key == "grad_norm") and not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {key}: {experiment}/{method}/{seed}")
    step_key = "global_step" if "global_step" in metrics else "env_step"
    returns = metrics["returned_episode_returns"]
    curves = []
    interval = cfg["LOG_INTERVAL"]
    for i in range(cfg["NUM_SEEDS"]):
        x, y = metrics[step_key][i], returns[i]
        keep = (metrics["update_steps"][i] % interval == 0) & np.isfinite(y)
        if experiment == "minatar":
            keep &= x <= 10_000_000
        curves.append(
            {
                "method": method,
                "env": cfg["ENV_NAME"],
                "seed": seed,
                "seed_index": i,
                "steps": x[keep].tolist(),
                "returns": y[keep].tolist(),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    name = f"{cfg['ENV_NAME'].split('-')[0].lower()}_{method}_{seed}.json"
    path = output / name
    # Atomic replacement lets interrupted suites resume safely.
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps({"config": cfg, "curves": curves}, separators=(",", ":")) + "\n"
    )
    temp.replace(path)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("minatar", "craftax"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two small updates per method; not paper results.",
    )
    parser.add_argument("--method", help="Run one method (e.g. gaussian or pqn).")
    parser.add_argument(
        "--run-index", type=int, help="Run one job from the selected suite."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing runs instead of resuming.",
    )
    args = parser.parse_args()
    selected = jobs(args.experiment, args.smoke)
    if args.method:
        selected = [(m, c) for m, c in selected if m == args.method]
        if not selected:
            parser.error(f"Unknown method: {args.method}")
    output = args.output.resolve() / ("smoke" if args.smoke else "") / args.experiment
    if args.run_index is not None:
        if not 0 <= args.run_index < len(selected):
            parser.error(f"--run-index must be between 0 and {len(selected) - 1}")
        method, cfg = selected[args.run_index]
        run(args.experiment, method, cfg, output)
        return
    for i, (method, cfg) in enumerate(selected):
        path = (
            output
            / f"{cfg['ENV_NAME'].split('-')[0].lower()}_{method}_{cfg['SEED']}.json"
        )
        if path.exists() and not args.force:
            saved = json.loads(path.read_text())
            if saved.get("config") != cfg:
                raise ValueError(
                    f"Settings changed for {path}; use a new --output or --force."
                )
            continue
        print(
            f"[{i + 1}/{len(selected)}] {args.experiment}: {method}, {cfg['ENV_NAME']}, seed {cfg['SEED']}",
            flush=True,
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            args.experiment,
            "--run-index",
            str(i),
            "--output",
            str(args.output.resolve()),
        ]
        if args.smoke:
            command.append("--smoke")
        if args.method:
            command.extend(["--method", args.method])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
