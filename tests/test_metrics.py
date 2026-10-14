import numpy as np

from src.metrics import decoding_accuracy, orientation_selectivity, rf_bar_scores, spike_counts
from src.stimulus import bar_image
from src.stimulus import Block


def test_spike_counts_ignore_rest_window_and_other_blocks():
    block = Block("b", start_ms=1000.0, thetas=np.zeros(2), offsets=np.zeros(2), plastic=False)
    spike_i = np.array([0, 1, 0, 1, 0])
    #                   pres0 pres0 rest  pres1 after block
    spike_t = np.array([1010, 1050, 1120, 1160, 1400]) - 0.0
    counts = spike_counts(spike_i, spike_t, 2, block, period_ms=150, present_ms=100)
    assert counts.tolist() == [[1, 1], [0, 1]]


def test_osi_extremes_and_preferred_orientation():
    tuning = np.array([
        [0, 0, 5, 0],   # only 90 deg
        [2, 2, 2, 2],   # untuned
        [0, 0, 0, 0],   # silent
    ], dtype=float)
    osi, pref = orientation_selectivity(tuning)
    assert np.isclose(osi[0], 1.0) and np.isclose(pref[0], np.pi / 2)
    assert np.isclose(osi[1], 0.0)
    assert np.isnan(osi[2]) and np.isnan(pref[2])


def test_osi_respects_responsive_mask():
    tuning = np.array([[0, 1, 0, 0], [0, 9, 0, 0]], dtype=float)
    osi, _ = orientation_selectivity(tuning, responsive=np.array([False, True]))
    assert np.isnan(osi[0]) and np.isclose(osi[1], 1.0)


def test_decoding_perfect_for_labelled_lines():
    n_or = 4
    theta_idx = np.tile(np.arange(n_or), 2)
    repeat_idx = np.repeat([0, 1], n_or)
    counts = np.eye(n_or)[theta_idx] * 5  # neuron k fires only for orientation k
    assert decoding_accuracy(counts, theta_idx, repeat_idx, n_or) == 1.0


def test_decoding_counts_silent_presentations_as_wrong():
    n_or = 2
    theta_idx = np.array([0, 1, 0, 1])
    repeat_idx = np.array([0, 0, 1, 1])
    counts = np.array([[3, 0], [0, 3], [0, 0], [0, 3]], dtype=float)
    assert decoding_accuracy(counts, theta_idx, repeat_idx, n_or) == 0.5


def test_rf_score_finds_bars_and_ignores_noise():
    size, rng = 16, np.random.default_rng(0)
    bar = bar_image(size, np.pi / 3, 1.0, 1.0).ravel()
    w = np.stack([bar + 0.05 * rng.random(size * size), rng.random(size * size)], axis=1)
    r = rf_bar_scores(w, size, width=1.0, offset_range_px=3.0)
    assert r[0] > 0.9 and r[1] < 0.3
