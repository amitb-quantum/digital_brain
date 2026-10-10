"""Orientation selectivity, decoding accuracy, and throughput metrics."""
import numpy as np


def spike_counts(spike_i, spike_t_ms, n_neurons, block, period_ms, present_ms):
    """Spike counts per (presentation, neuron) during the stimulus-on window
    of each presentation in `block`."""
    rel = spike_t_ms - block.start_ms
    pres = np.floor(rel / period_ms).astype(int)
    keep = (rel >= 0) & (pres < block.n) & (rel - pres * period_ms < present_ms)
    counts = np.zeros((block.n, n_neurons))
    np.add.at(counts, (pres[keep], spike_i[keep]), 1)
    return counts


def tuning_curves(counts, theta_idx, n_orientations):
    """Mean spike count per (neuron, orientation bin)."""
    return np.stack(
        [counts[theta_idx == k].mean(axis=0) for k in range(n_orientations)], axis=1
    )


def orientation_selectivity(tuning, responsive=None):
    """Circular-variance OSI = |sum r e^{2i theta}| / sum r (0 = untuned,
    1 = responds to one orientation only) and preferred orientation in
    radians [0, pi). Neurons outside `responsive` (default: any spike) get NaN.

    OSI is biased upward for neurons with very few spikes (one spike gives
    OSI = 1), so callers should pass a minimum-spike mask."""
    n_or = tuning.shape[1]
    thetas = np.arange(n_or) * np.pi / n_or
    vec = tuning @ np.exp(2j * thetas)
    total = tuning.sum(axis=1)
    if responsive is None:
        responsive = total > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = np.where(responsive, np.abs(vec) / total, np.nan)
    pref = np.where(responsive, np.mod(np.angle(vec), 2 * np.pi) / 2, np.nan)
    return osi, pref


def decoding_accuracy(counts, theta_idx, repeat_idx, n_orientations):
    """Diehl & Cook style readout: label each neuron with the orientation it
    fires most for on repeat 0, then classify later repeats by which label
    group has the highest mean response. Chance is 1 / n_orientations."""
    label_set = repeat_idx == 0
    eval_set = ~label_set
    if not eval_set.any():
        return float("nan")
    tuning = tuning_curves(counts[label_set], theta_idx[label_set], n_orientations)
    active = tuning.sum(axis=1) > 0
    labels = np.where(active, tuning.argmax(axis=1), -1)

    scores = np.full((eval_set.sum(), n_orientations), -np.inf)
    for k in range(n_orientations):
        members = labels == k
        if members.any():
            scores[:, k] = counts[eval_set][:, members].mean(axis=1)
    silent = counts[eval_set].sum(axis=1) == 0
    predicted = np.where(silent, -1, scores.argmax(axis=1))
    return float(np.mean(predicted == theta_idx[eval_set]))


def evaluate_block(spike_i, spike_t_ms, n_exc, block, schedule, n_orientations,
                   min_spikes):
    counts = spike_counts(spike_i, spike_t_ms, n_exc, block,
                          schedule.period_ms, schedule.present_ms)
    tuning = tuning_curves(counts, block.theta_idx, n_orientations)
    responsive = counts.sum(axis=0) >= min_spikes
    osi, pref = orientation_selectivity(tuning, responsive)
    return {
        "summary": {
            "responsive_fraction": float(responsive.mean()),
            "mean_osi": float(np.nanmean(osi)) if responsive.any() else float("nan"),
            "median_osi": float(np.nanmedian(osi)) if responsive.any() else float("nan"),
            "decoding_accuracy": decoding_accuracy(
                counts, block.theta_idx, block.repeat_idx, n_orientations),
            "chance_accuracy": 1.0 / n_orientations,
            "mean_spikes_per_presentation": float(counts.sum(axis=1).mean()),
        },
        "tuning": tuning,
        "osi": osi,
        "pref": pref,
    }


def throughput(results, n_exc):
    """Simulation speed relative to biology, plus synaptic events per second.

    real_time_factor > 1 means the model runs faster than the tissue it
    emulates. Synaptic events count every delivered spike per target synapse.
    """
    t = results["timing"]
    syn_events = (results["n_input_spikes"] * n_exc
                  + results["n_exc_spikes"]
                  + results["n_inh_spikes"] * (n_exc - 1))
    return {
        **t,
        "real_time_factor": t["bio_time_s"] / t["sim_wall_s"],
        "synaptic_events": int(syn_events),
        "synaptic_events_per_wall_s": syn_events / t["sim_wall_s"],
        "n_input_spikes": results["n_input_spikes"],
        "n_exc_spikes": results["n_exc_spikes"],
        "n_inh_spikes": results["n_inh_spikes"],
    }
