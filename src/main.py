"""Run the V1 orientation-learning experiment for one compute tier.

    python -m src.main --config config/laptop.yaml
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from . import plots
from .config import load_config
from .metrics import evaluate_block, learned_rfs, throughput
from .network import initial_weights, run_simulation
from .stimulus import build_schedule


def run_experiment(cfg, out_root=None, make_plots=True, keep_build=False):
    out_root = Path(out_root or cfg["output_dir"])
    run_dir = out_root / f"{cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rng = np.random.default_rng(cfg["seed"])
    schedule = build_schedule(cfg, rng)
    w_init = initial_weights(cfg, rng)

    build_dir = run_dir / "build"
    results = run_simulation(cfg, schedule, w_init, build_dir)
    if not keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)

    n_exc = cfg["network"]["n_exc"]
    n_or = cfg["test"]["n_orientations"]
    evals = {
        name: evaluate_block(results["spike_i"], results["spike_t_ms"], n_exc,
                             schedule.blocks[name], schedule, n_or,
                             cfg["test"]["min_spikes"])
        for name in ("pre_test", "post_test")
    }
    metrics = {
        "config": cfg["name"],
        "device": cfg["backend"]["device"],
        "n_inputs": int(w_init.shape[0]),
        "n_exc": n_exc,
        "n_synapses": int(w_init.size + n_exc + n_exc * (n_exc - 1)),
        "pre_test": evals["pre_test"]["summary"],
        "post_test": evals["post_test"]["summary"],
        "learned_rfs": {"initial": learned_rfs(w_init, cfg),
                        "final": learned_rfs(results["w_final"], cfg)},
        "performance": throughput(results, n_exc),
        "theta_mean_mV": float(results["theta_final_mV"].mean()),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(run_dir / "results.npz", w_init=w_init, w_final=results["w_final"],
                        theta_final_mV=results["theta_final_mV"],
                        spike_i=results["spike_i"], spike_t_ms=results["spike_t_ms"])

    if make_plots:
        size = cfg["stimulus"]["size"]
        plots.receptive_fields(w_init, size, run_dir / "rf_initial.png", "Input weights before training")
        plots.receptive_fields(results["w_final"], size, run_dir / "rf_final.png", "Input weights after training")
        plots.selectivity(evals["pre_test"], evals["post_test"], run_dir / "selectivity.png")

    return run_dir, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/laptop.yaml")
    parser.add_argument("--out", default=None, help="output root (default: config output_dir)")
    parser.add_argument("--keep-build", action="store_true", help="keep generated standalone code")
    parser.add_argument("--device", default=None,
                        help="override backend.device, e.g. run a GPU-sized config on the CPU")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["backend"]["device"] = args.device
        cfg["name"] = f"{cfg['name']}-{args.device}"
    run_dir, metrics = run_experiment(cfg, args.out, keep_build=args.keep_build)
    print(json.dumps({k: metrics[k] for k in ("pre_test", "post_test", "learned_rfs", "performance")}, indent=2))
    print(f"\nResults in {run_dir}")


if __name__ == "__main__":
    main()
