import numpy as np

from src.config import load_config
from src.stimulus import bar_image, build_schedule


def test_horizontal_bar_peaks_on_centre_row():
    img = bar_image(15, theta=0.0, offset=0.0, width=1.0)
    assert np.argmax(img.sum(axis=1)) == 7


def test_vertical_bar_peaks_on_centre_column():
    img = bar_image(15, theta=np.pi / 2, offset=0.0, width=1.0)
    assert np.argmax(img.sum(axis=0)) == 7


def test_offset_shifts_bar_along_normal():
    img = bar_image(15, theta=0.0, offset=3.0, width=1.0)
    assert np.argmax(img.sum(axis=1)) == 10


def test_aperture_blanks_corners():
    img = bar_image(16, theta=np.pi / 4, offset=0.0, width=1.0)
    assert img[0, 0] == 0 and img[-1, -1] == 0


def test_schedule_layout():
    cfg = load_config("config/smoke.yaml")
    sched = build_schedule(cfg, np.random.default_rng(0))
    n_test = cfg["test"]["n_orientations"] * len(cfg["test"]["offsets_px"]) * cfg["test"]["repeats"]
    n_total = 2 * n_test + cfg["training"]["n_presentations"]
    assert sched.duration_ms == n_total * sched.period_ms

    train = sched.blocks["train"]
    assert train.start_ms == n_test * sched.period_ms
    rows_per_pres = int(sched.period_ms / sched.chunk_ms)
    first = int(train.start_ms / sched.chunk_ms)
    last = first + train.n * rows_per_pres
    assert sched.plastic[first:last].all()
    assert not sched.plastic[:first].any() and not sched.plastic[last:].any()

    # Rest rows (after the stimulus-on rows of each presentation) are silent.
    rows_on = int(sched.present_ms / sched.chunk_ms)
    per_pres = sched.rates_hz.reshape(n_total, rows_per_pres, -1)
    assert (per_pres[:, rows_on:] == 0).all()
    assert (per_pres[:, :rows_on].max(axis=2) > 0).all()


def test_test_blocks_cover_every_orientation_each_repeat():
    cfg = load_config("config/smoke.yaml")
    block = build_schedule(cfg, np.random.default_rng(0)).blocks["post_test"]
    for r in range(cfg["test"]["repeats"]):
        assert sorted(set(block.theta_idx[block.repeat_idx == r])) == list(range(cfg["test"]["n_orientations"]))
    # repeat 0 is presented entirely before repeat 1
    assert (np.diff(block.repeat_idx) >= 0).all()
