"""Oriented-bar stimuli and the presentation schedule that drives the network.

The whole experiment (pre-test, training, post-test) is laid out ahead of time
as a rate table plus a plasticity on/off signal, so it runs as a single
simulation call. That keeps it compatible with the standalone devices
(cpp_standalone, cuda_standalone), which cannot loop in Python between runs.
"""
from dataclasses import dataclass, field
from math import gcd

import numpy as np


def bar_image(size, theta, offset, width):
    """Bar through the retina at orientation `theta` (radians), shifted
    `offset` pixels along its normal. Pixels outside the inscribed circle are
    zero so no orientation gets longer bars from the square's corners."""
    c = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size] - c
    dist = -x * np.sin(theta) + y * np.cos(theta) - offset
    img = np.exp(-dist**2 / (2 * width**2))
    img[x**2 + y**2 > (c + 0.5) ** 2] = 0.0
    return img


@dataclass
class Block:
    name: str
    start_ms: float
    thetas: np.ndarray
    offsets: np.ndarray
    plastic: bool
    theta_idx: np.ndarray = None  # orientation bin per presentation (test blocks)
    repeat_idx: np.ndarray = None

    @property
    def n(self):
        return len(self.thetas)


@dataclass
class Schedule:
    rates_hz: np.ndarray    # (n_chunks, n_inputs)
    plastic: np.ndarray     # (n_chunks,) 1.0 while STDP is on
    chunk_ms: float
    present_ms: float
    period_ms: float
    blocks: dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        return len(self.plastic) * self.chunk_ms


def _test_block(name, cfg, rng):
    n_or = cfg["test"]["n_orientations"]
    grid = [
        (k, off, r)
        for r in range(cfg["test"]["repeats"])
        for k in range(n_or)
        for off in cfg["test"]["offsets_px"]
    ]
    # Shuffle within each repeat so repeat 0 is fully presented before repeat 1.
    order = np.concatenate(
        [rng.permutation(np.flatnonzero([g[2] == r for g in grid]))
         for r in range(cfg["test"]["repeats"])]
    )
    grid = [grid[i] for i in order]
    theta_idx = np.array([g[0] for g in grid])
    return Block(
        name=name,
        start_ms=0.0,
        thetas=theta_idx * np.pi / n_or,
        offsets=np.array([g[1] for g in grid], dtype=float),
        plastic=False,
        theta_idx=theta_idx,
        repeat_idx=np.array([g[2] for g in grid]),
    )


def build_schedule(cfg, rng):
    s = cfg["stimulus"]
    present, rest = int(s["present_ms"]), int(s["rest_ms"])
    chunk = gcd(present, rest) if rest else present
    rows_on, rows_off = present // chunk, rest // chunk

    n_train = cfg["training"]["n_presentations"]
    train = Block(
        name="train",
        start_ms=0.0,
        thetas=rng.uniform(0, np.pi, n_train),
        offsets=rng.uniform(-s["offset_range_px"], s["offset_range_px"], n_train),
        plastic=True,
    )
    blocks = [_test_block("pre_test", cfg, rng), train, _test_block("post_test", cfg, rng)]

    rates, plastic = [], []
    t = 0.0
    silence = np.zeros((rows_off, s["size"] ** 2))
    for block in blocks:
        block.start_ms = t
        for theta, offset in zip(block.thetas, block.offsets):
            img = bar_image(s["size"], theta, offset, s["bar_width_px"])
            rates.append(np.repeat(img.reshape(1, -1) * s["max_rate_hz"], rows_on, axis=0))
            rates.append(silence)
            plastic.extend([float(block.plastic)] * (rows_on + rows_off))
            t += present + rest

    return Schedule(
        rates_hz=np.concatenate(rates),
        plastic=np.array(plastic),
        chunk_ms=float(chunk),
        present_ms=float(present),
        period_ms=float(present + rest),
        blocks={b.name: b for b in blocks},
    )
