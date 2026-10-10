"""Spiking V1-style circuit: Poisson retina -> excitatory LIF layer (triplet
STDP, adaptive thresholds) <-> inhibitory layer (winner-take-all).

The same model code runs on every compute tier; only `backend.device` in the
config changes (runtime / cpp_standalone on CPU, cuda_standalone on GPU).
"""
import time
from pathlib import Path

import brian2 as b2
import numpy as np
from brian2 import Hz, mV, ms

EXC_EQS = """
dv/dt = ((v_rest - v) + ge * (E_exc - v) + gi * (E_inh - v)) / tau_m : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
dtheta/dt = -theta / tau_theta : volt
w_total : 1
"""

INH_EQS = """
dv/dt = ((v_rest - v) + ge * (E_exc - v) + gi * (E_inh - v)) / tau_m : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
"""

# Triplet STDP (Pfister & Gerstner 2006, as used by Diehl & Cook 2015), gated
# by plastic(t) so the same synapses are frozen during the test blocks.
STDP_MODEL = """
w : 1
dapre/dt = -apre / tau_pre : 1 (event-driven)
dapost1/dt = -apost1 / tau_post1 : 1 (event-driven)
dapost2/dt = -apost2 / tau_post2 : 1 (event-driven)
w_total_post = w : 1 (summed)
"""
STDP_ON_PRE = """
ge_post += w
apre = 1.0
w = clip(w - plastic(t) * nu_pre * apost1, 0, wmax)
"""
STDP_ON_POST = """
w = clip(w + plastic(t) * nu_post * apre * apost2, 0, wmax)
apost1 = 1.0
apost2 = 1.0
"""


def _neuron_namespace(p):
    return {
        "v_rest": p["v_rest_mV"] * mV,
        "v_reset": p["v_reset_mV"] * mV,
        "v_thresh": p["v_thresh_mV"] * mV,
        "tau_m": p["tau_m_ms"] * ms,
        "E_exc": p["E_exc_mV"] * mV,
        "E_inh": p["E_inh_mV"] * mV,
        "tau_ge": p["tau_ge_ms"] * ms,
        "tau_gi": p["tau_gi_ms"] * ms,
    }


def configure_device(backend, build_dir):
    """Select the Brian2 device. Must run before any Brian2 object is created."""
    device = backend["device"]
    b2.device.reinit()
    b2.device.activate()
    if device == "runtime":
        b2.set_device("runtime")
        b2.prefs.codegen.target = backend.get("codegen", "numpy")
    elif device == "cpp_standalone":
        b2.set_device("cpp_standalone", directory=str(build_dir), build_on_run=True)
        b2.prefs.devices.cpp_standalone.openmp_threads = backend.get("threads", 1)
    elif device == "cuda_standalone":
        import brian2cuda  # noqa: F401  registers the device
        b2.set_device("cuda_standalone", directory=str(build_dir), build_on_run=True)
    else:
        raise ValueError(f"unknown backend.device {device!r}")
    b2.defaultclock.dt = backend["dt_ms"] * ms


def initial_weights(cfg, rng):
    n_in = cfg["stimulus"]["size"] ** 2
    n_exc = cfg["network"]["n_exc"]
    return (rng.random((n_in, n_exc)) + 0.01) * cfg["plasticity"]["w_init_max"]


def run_simulation(cfg, schedule, w_init, build_dir):
    """Build and run the full pre-test/train/post-test timeline once.

    Returns plain numpy results plus wall-clock timings, so callers never touch
    Brian2 objects (whose values are device-dependent until the run finishes).
    """
    configure_device(cfg["backend"], Path(build_dir))
    b2.seed(cfg["seed"])

    net_cfg, plast = cfg["network"], cfg["plasticity"]
    n_in, n_exc = w_init.shape

    stim = b2.TimedArray(schedule.rates_hz * Hz, dt=schedule.chunk_ms * ms)
    plastic = b2.TimedArray(schedule.plastic, dt=schedule.chunk_ms * ms)

    inp = b2.PoissonGroup(n_in, rates="stim(t, i)", name="retina")

    exc_p = net_cfg["exc"]
    exc_ns = _neuron_namespace(exc_p) | {
        "theta_plus": exc_p["theta_plus_mV"] * mV,
        "tau_theta": exc_p["tau_theta_ms"] * ms,
        "plastic": plastic,
    }
    exc = b2.NeuronGroup(
        n_exc, EXC_EQS, threshold="v > v_thresh + theta",
        reset="v = v_reset; theta += plastic(t) * theta_plus",
        refractory=exc_p["refractory_ms"] * ms, method="euler",
        namespace=exc_ns, name="exc",
    )
    exc.v = exc_p["v_rest_mV"] * mV

    inh_p = net_cfg["inh"]
    inh = b2.NeuronGroup(
        n_exc, INH_EQS, threshold="v > v_thresh", reset="v = v_reset",
        refractory=inh_p["refractory_ms"] * ms, method="euler",
        namespace=_neuron_namespace(inh_p), name="inh",
    )
    inh.v = inh_p["v_rest_mV"] * mV

    syn_ns = {
        "tau_pre": plast["tau_pre_ms"] * ms,
        "tau_post1": plast["tau_post1_ms"] * ms,
        "tau_post2": plast["tau_post2_ms"] * ms,
        "nu_pre": plast["nu_pre"],
        "nu_post": plast["nu_post"],
        "wmax": plast["wmax"],
        "w_target": plast["norm_mean_w"] * n_in,
        "plastic": plastic,
    }
    in_exc = b2.Synapses(inp, exc, STDP_MODEL, on_pre=STDP_ON_PRE, on_post=STDP_ON_POST,
                         method="exact", namespace=syn_ns, name="in_exc")
    pre_idx, post_idx = np.meshgrid(np.arange(n_in), np.arange(n_exc), indexing="ij")
    in_exc.connect(i=pre_idx.ravel(), j=post_idx.ravel())
    in_exc.w = w_init.ravel()
    # Synaptic scaling: at each stimulus onset during training, rescale every
    # neuron's incoming weights to a fixed total so no neuron monopolises input.
    in_exc.run_regularly(
        "w = w * (1 + plastic(t) * (w_target / (w_total_post + 1e-12) - 1))",
        dt=schedule.period_ms * ms, when="start",
    )

    exc_inh = b2.Synapses(exc, inh, on_pre="ge_post += w_ei",
                          namespace={"w_ei": net_cfg["w_exc_inh"]}, name="exc_inh")
    exc_inh.connect(j="i")
    inh_exc = b2.Synapses(inh, exc, on_pre="gi_post += w_ie",
                          namespace={"w_ie": net_cfg["w_inh_exc"]}, name="inh_exc")
    inh_exc.connect(condition="i != j")

    exc_mon = b2.SpikeMonitor(exc, name="exc_spikes")
    inp_mon = b2.SpikeMonitor(inp, record=False, name="inp_spikes")
    inh_mon = b2.SpikeMonitor(inh, record=False, name="inh_spikes")

    net = b2.Network(inp, exc, inh, in_exc, exc_inh, inh_exc, exc_mon, inp_mon, inh_mon)

    t0 = time.perf_counter()
    net.run(schedule.duration_ms * ms)
    total_wall = time.perf_counter() - t0

    if cfg["backend"]["device"] == "runtime":
        sim_wall, compile_wall = total_wall, None
    else:
        timers = b2.device.timers
        sim_wall = timers["run_binary"]
        compile_wall = timers["compile"]["make"]

    w_final = np.zeros((n_in, n_exc))
    w_final[np.asarray(in_exc.i[:]), np.asarray(in_exc.j[:])] = np.asarray(in_exc.w[:])

    return {
        "spike_i": np.asarray(exc_mon.i[:]),
        "spike_t_ms": np.asarray(exc_mon.t[:] / ms),
        "n_input_spikes": int(inp_mon.num_spikes),
        "n_exc_spikes": int(exc_mon.num_spikes),
        "n_inh_spikes": int(inh_mon.num_spikes),
        "w_final": w_final,
        "theta_final_mV": np.asarray(exc.theta[:] / mV),
        "timing": {
            "bio_time_s": schedule.duration_ms / 1000,
            "sim_wall_s": sim_wall,
            "compile_wall_s": compile_wall,
            "total_wall_s": total_wall,
        },
    }
