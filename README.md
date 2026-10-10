# digital_brain

Proof of concept that a single GPU can **emulate** part of the brain: biologically
modelled spiking neurons, conductance-based membranes and spike-timing-dependent
plasticity (STDP), not a conventional deep-learning network. For the motivation
(Geoffrey Hinton on digital vs. biological intelligence) and the reasoning behind
the design, see [docs/background.md](docs/background.md).

## First circuit: V1 orientation learning

A primary-visual-cortex-style microcircuit learns orientation selectivity from
scratch, with no labels and no backprop:

```
retina (N×N Poisson inputs) ──STDP──▶ excitatory LIF layer ──1:1──▶ inhibitory layer
                                            ▲                              │
                                            └──── lateral inhibition ◀─────┘
```

- **Neurons:** conductance-based leaky integrate-and-fire neurons. Excitatory neurons have adaptive thresholds.
- **Learning:** triplet STDP on the input synapses, plus synaptic scaling that keeps each neuron's total input fixed.
- **Competition:** winner-take-all via the inhibitory layer (architecture after Diehl & Cook 2015).
- **Stimulus:** oriented bars at random angles and positions.

Each run has three blocks, simulated in one call: **pre-test** (plasticity off,
random weights as the control), **training** (plasticity on), and **post-test**
(plasticity off).

**Metrics reported**
- Orientation selectivity index (OSI). Only neurons with at least `test.min_spikes` spikes count, because OSI is inflated for neurons that barely fire.
- Population decoding accuracy (chance = 1 / n_orientations).
- Real-time factor (biological seconds simulated per wall-clock second).
- Synaptic events per second.

## Compute tiers

The same model code runs everywhere. Only the config file changes.

| Config | Hardware | Brian2 device | Status |
|---|---|---|---|
| `config/smoke.yaml` | any | `runtime` (numpy) | tests, seconds |
| `config/laptop.yaml` | Dell, Core Ultra 9 288V, WSL2 | `cpp_standalone` | validated |
| `config/rtx5090.yaml` | RTX 5090 desktop (32 GB) | `cuda_standalone` | needs `brian2cuda` and CUDA ≥ 12.8 |
| `config/vast_h200.yaml` | vast.ai H100/H200 | `cuda_standalone` | not yet validated |

`config/default.yaml` holds every parameter. Tier files override only what differs.

## Usage

```bash
conda activate digital-brain
pip install -r requirements.txt
python -m src.main --config config/laptop.yaml
pytest
```

Each run writes `data/runs/<config>-<timestamp>/` containing `metrics.json`,
`config.yaml`, `results.npz`, receptive-field images (`rf_initial.png`,
`rf_final.png`) and `selectivity.png`.

## Notes

- **NumPy is pinned below 2.4.** Brian2 2.9.0 still calls `ndarray.ptp`, which NumPy 2.4 removed.
- **Use one thread on the laptop.** On this 25k-synapse network, `cpp_standalone` with 1 OpenMP thread ran 16× faster than with 8, because thread synchronisation per timestep costs more than the work. Re-check this as the network grows.
- **Keep secrets in `.env`**, which is git-ignored. Example: `VAST_API_KEY`.
