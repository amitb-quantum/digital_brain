# CLAUDE.md

Context for Claude Code / Cowork sessions on any machine. Read this,
`README.md`, and `docs/background.md` before starting work.

## What this project is

A proof of concept that a single GPU can **emulate** part of the brain using
brain-like mechanisms: spiking neurons, conductance-based membranes, and STDP
learning. It is deliberately **not** a conventional ANN/LLM. Keep designs
spiking and biophysical. The motivation (Hinton on digital vs. biological
intelligence) and the design reasoning are in `docs/background.md`.

The first circuit is V1 orientation learning:
- **Input:** a Poisson retina.
- **Excitatory layer:** LIF neurons with triplet STDP, adaptive thresholds and synaptic scaling.
- **Inhibitory layer:** winner-take-all competition.
- **Basis:** Diehl & Cook 2015.

Candidates for later circuits: the hippocampus (CA3/CA1, pattern
completion) and a cerebellar microcircuit (motor control).

## Layout

- `src/network.py`: the Brian2 model. `configure_device()` selects the backend.
- `src/stimulus.py`: precomputes the whole pre-test → train → post-test timeline as `TimedArray`s, so a run is **one `net.run()` call**. Standalone devices (cpp/cuda) need this because they cannot loop in Python. Keep it that way.
- `src/metrics.py`: OSI (filtered by `test.min_spikes`), decoding accuracy, real-time factor, and synaptic events per second.
- `src/main.py`: `python -m src.main --config config/<tier>.yaml`. Writes `data/runs/<name>-<timestamp>/`, which git ignores.
- `config/default.yaml` holds every parameter. Tier files deep-merge on top and override only what differs.
- `tests/`: `pytest` (about 3 s; the smoke test runs the tiny numpy-runtime network).

## Environment

- **Conda env `digital-brain`, Python 3.11:** `pip install -r requirements.txt`.
- **NumPy must stay below 2.4.** Brian2 2.9.0 calls `ndarray.ptp`, which NumPy 2.4 removed, and Brian2 fails at import.
- **YAML gotcha:** PyYAML reads `1.0e7` as a string. Write `1.0e+7`.

## Compute tiers

The same model code runs on every tier. Only the config changes.

| Config | Hardware | Device | Status |
|---|---|---|---|
| `smoke.yaml` | any | `runtime` (numpy) | used by tests |
| `laptop.yaml` | Laptop, Intel Core Ultra 9 288V, WSL2, no NVIDIA GPU | `cpp_standalone`, 1 thread | **validated** |
| `rtx5090.yaml` | Desktop "vermithor", RTX 5090 32 GB | `cuda_standalone` | **next step, untested** |
| `vast_h200.yaml` | rented vast.ai H100/H200 | `cuda_standalone` | untested |

**Laptop reference run** (`config/laptop.yaml`, seed 42, fully deterministic):

| Metric | Result |
|---|---|
| Decoding accuracy (chance 0.125) | 0.375 before → 0.875 after |
| Mean OSI | 0.65 → 0.93 |
| Responsive fraction in post-test | 0.17 |
| Speed | 558.6 s biological in 18.6 s wall, real-time factor about 30 |

About 50 of the 100 excitatory neurons learn clean oriented-bar receptive fields.

### Lessons from tuning

- **OpenMP threads:** on the 25k-synapse network, 1 thread was 16× faster than 8, because per-timestep sync overhead dominates. Re-measure when the network grows.
- **Input rate:** `max_rate_hz` 64 left neurons near-silent. 192 works.
- **Threshold adaptation:** `theta_plus_mV` above 0.05 (tried 0.2 and 0.5) silences the network at test time. Keep 0.05.
- **Low responsive fraction:** partly by design (winner-take-all, sparse tuning), partly because about half the neurons are never recruited. Open problem: try longer training or more neurons.

## Next step: bring up the RTX 5090 tier

1. **Check the GPU.** `nvidia-smi` must show the 5090.
   - On WSL2, the driver comes from Windows. Do **not** install a Linux NVIDIA driver inside WSL. Install only the CUDA toolkit (the WSL-Ubuntu package).
2. **Install a CUDA toolkit of 12.8 or newer.** Blackwell is compute capability 12.0 (`sm_120`), and older toolkits cannot target it. `nvcc --version` must work.
3. **Install `brian2cuda`** into the `digital-brain` env.
   - Check that it accepts Brian2 2.9.0 and NumPy below 2.4 without upgrading NumPy.
   - If it cannot detect or target compute capability 12.0, look at the `prefs.devices.cuda_standalone.cuda_backend.*` preferences, such as the compute capability and the nvcc path.
4. **Validate before scaling.**
   - First run the laptop-sized model with `device: cuda_standalone`. Metrics should be close to the CPU reference above, but not bit-identical, because the RNG streams differ.
   - Only then run `config/rtx5090.yaml` (32×32 retina, 1600 excitatory neurons, 5000 presentations).
5. **Record the results:** real-time factor and synaptic events per second for each size. These are the PoC's headline numbers.

## Rules

- **The repo `github.com/amitb-quantum/digital_brain` is PRIVATE.** Do not change its visibility; the owner will decide when to make it public.
- **Never commit `digitalbrain.pdf`** (it's in `.gitignore`). It's the original source conversation. `docs/background.md` is its sanitised replacement. Keep that doc free of chat transcripts, chat links, usernames, hostnames and account details.
- **Secrets** (for example `VAST_API_KEY`) go in `.env`, which git ignores, or in `vastai set api-key`. Never put them in code or config.
- **vast.ai instances must be destroyed when a run finishes.** An idle H200 costs about $50 a day. Any launch script must tear the instance down.
- **Report results honestly.** Treat the claims in `docs/background.md` §4 (speedups, capacity, "perfect memory") as hypotheses to measure, not facts.
