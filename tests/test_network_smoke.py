"""End-to-end run of the tiny smoke config on the numpy runtime (~seconds)."""
import json

import numpy as np

from src.config import load_config
from src.main import run_experiment


def test_smoke_experiment_runs_and_reports(tmp_path):
    cfg = load_config("config/smoke.yaml")
    cfg["test"]["min_spikes"] = 1
    run_dir, metrics = run_experiment(cfg, out_root=tmp_path, make_plots=False)

    perf = metrics["performance"]
    assert perf["n_input_spikes"] > 0
    assert perf["n_exc_spikes"] > 0, "excitatory layer never fired"
    assert perf["n_inh_spikes"] > 0
    assert perf["real_time_factor"] > 0
    for block in ("pre_test", "post_test"):
        assert 0.0 <= metrics[block]["decoding_accuracy"] <= 1.0
    assert json.loads((run_dir / "metrics.json").read_text())["config"] == "smoke"

    data = np.load(run_dir / "results.npz")
    w = data["w_final"]
    assert w.min() >= 0 and w.max() <= cfg["plasticity"]["wmax"]
    # Synaptic scaling keeps each neuron's total input near the target.
    target = cfg["plasticity"]["norm_mean_w"] * w.shape[0]
    assert np.allclose(w.sum(axis=0), target, rtol=0.2)
