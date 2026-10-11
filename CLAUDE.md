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

- **Conda env `digital-brain`, Python 3.12:** `pip install -r requirements.txt`.
- **Brian2 is pinned to 2.10.1 because brian2cuda pins it exactly.** brian2cuda 1.0b1 needs Brian2 2.10.1, which needs Python 3.12 or newer; 1.0a7 needs 2.8.0. Upgrade Brian2 and brian2cuda together. Brian2 2.10 fixed the `ndarray.ptp` problem, so NumPy is no longer held below 2.4.
- **Copy Brian2 arrays out of a run with `np.array`, never `np.asarray`.** Since Brian2 2.10, monitor arrays such as `mon.i[:]` are views into memory that is freed with the network. A view kept past `run_simulation` returns garbage.
- **CUDA toolkit for the GPU tiers comes from conda-forge into the env** (no sudo needed):
  `conda install -n digital-brain -c conda-forge "cuda-version=12.9" cuda-nvcc cuda-cudart-dev libcurand-dev cuda-profiler-api`.
  This also installs conda's gcc 14, because CUDA 12.9 does not support gcc 15.
  - Run through `conda activate` or `conda run`, so that `$CXX` points at that gcc.
  - Avoid CUDA 13: brian2cuda builds with `-std=c++11`, and CUDA 13's CCCL needs C++17.
  - `configure_device()` appends `-std=c++17` to the nvcc flags, because gcc 14's headers do not parse as C++11. nvcc keeps the last `-std`, so its "incompatible redefinition" warning is expected.
- **YAML gotcha:** PyYAML reads `1.0e7` as a string. Write `1.0e+7`.

## Compute tiers

The same model code runs on every tier. Only the config changes.

| Config | Hardware | Device | Status |
|---|---|---|---|
| `smoke.yaml` | any | `runtime` (numpy) | used by tests |
| `laptop.yaml` | Laptop, Intel Core Ultra 9 288V, WSL2, no NVIDIA GPU | `cpp_standalone`, 1 thread | **validated** |
| `rtx5090_validate.yaml` | Desktop "vermithor", RTX 5090 32 GB, WSL2 | `cuda_standalone` | **validated**: learns, close to the CPU reference |
| `rtx5090.yaml` | same | `cuda_standalone` | **runs**; learning needs retuning at this size |
| `vast_h200.yaml` | rented vast.ai H100/H200 | `cuda_standalone` | untested on an H200; a short run at this size works on the 5090 |

**Integrator.** `network.method` defaults to `exponential_euler`. The first full `rtx5090.yaml` run used `euler` and failed. At dt 0.5 ms, forward Euler goes unstable once many inhibitory neurons spike in one step: `gi` jumps past ~400 and `v` overshoots past E_inh. Every neuron then fired at about 180 Hz, even before training. The CPU showed the same, so it is not a GPU bug. dt 0.1 ms also fixes it, but is 5× slower.

**Laptop-size reference** (`config/laptop.yaml`, seed 42, deterministic):

| Metric | `euler` (old reference) | `exponential_euler` (current default) |
|---|---|---|
| Decoding accuracy (chance 0.125) | 0.375 → 0.875 | 0.375 → 0.708 |
| Post-test accuracy, mean over seeds 42, 1, 2, 3 | 0.917 | 0.81 |
| Mean OSI | 0.65 → 0.93 | 0.65 → 0.935 |
| Responsive fraction in post-test | 0.17 | 0.15 |
| Real-time factor, laptop CPU | about 30 | not measured |
| Real-time factor, vermithor CPU | about 40 | about 32 |

Under `exponential_euler`, decoding is about 0.1 lower and OSI is unchanged. The parameters were hand-tuned under `euler`, which at dt/τ_ge = 0.5 decays conductances faster than the exact solution. Retuning is an open task.

About 50 of the 100 excitatory neurons learn clean oriented-bar receptive fields.

### RTX 5090 results (2026-10-10, WSL2, CUDA 12.9, brian2cuda 1.0b1)

| Model size | Synapses | GPU RTF | CPU RTF, 1 thread (vermithor) | GPU µs per 0.5 ms step |
|---|---|---|---|---|
| 16×16 retina, 100 exc (`rtx5090_validate`) | 36k | 1.24 | about 32 to 40 | about 400 |
| 32×32 retina, 1600 exc (`rtx5090`) | 4.2M | 1.16 (full run) | 0.44 (8 threads: 0.46) | about 430 |
| 64×64 retina, 6400 exc (`vast_h200` size, short run) | 67M | 0.47 | not run | about 1070 |

- **The GPU is latency-bound, not compute-bound, up to about 4M synapses.** Per-step cost stays flat from 36k to 4.2M synapses, and grows only 2.5× for 16× more synapses after that. At 4.2M synapses the GPU is 2.4× faster than the CPU. At laptop size it is about 30× slower.
- **The cause is WSL2 sync latency.** A minimal kernel test measured a 7 µs launch, a 23 µs sync and a 77 µs 4-byte device-to-host copy, several times native Linux. brian2cuda makes about 7 such copies per step: 3 spike monitors, plus 4 synaptic pathways that read their spike counts to size the kernel launch.
- **Full `rtx5090.yaml` run:** 1784 s of biological time in 1534 s wall time, plus 27 s compile, at 2.1×10⁷ synaptic events per second.
  - Decoding accuracy went 0.79 → 0.875 and mean OSI 0.74 → 0.98.
  - Only 3% of neurons respond in the post-test. Random weights already decode at 0.79 at this size, so learning adds little.
- **Short run at `vast_h200` size on the 5090:** about 7.5×10⁷ synaptic events per second. The 5090 has room for this size, and a full run would take about 4 hours locally.

### Lessons from tuning

- **OpenMP threads:** on the 25k-synapse network, 1 thread was 16× faster than 8, because per-timestep sync overhead dominates. At 4.2M synapses, 8 threads were only 3% faster than 1.
- **Input rate:** `max_rate_hz` 64 left neurons near-silent. 192 works at 16×16. At larger retinas, bars are longer and drive is higher: 9% of neurons respond in the pre-test at 32×32 and 71% at 64×64.
- **Threshold adaptation:** `theta_plus_mV` above 0.05 (tried 0.2 and 0.5) silences the network at test time. Keep 0.05.
- **Low responsive fraction:** partly by design (winner-take-all, sparse tuning), partly because many neurons are never recruited. At 1600 neurons only 3% respond after training. This is the main open problem.

## Next steps

1. **Retune for scale.** Make the learning work at 32×32 with 1600 neurons before running anything bigger.
   - Scale input drive with retina size, for example `max_rate_hz` or `norm_mean_w`, so pre-test activity stays sparse.
   - Recruit more neurons: longer training, or a larger `theta_plus` with fewer neurons.
   - Retune laptop size under `exponential_euler` to recover the about 0.1 decoding gap.
2. **Cut per-step GPU overhead.**
   - Drop the input and inhibitory `SpikeMonitor`s, which are used only for counts, and count spikes with a variable in the reset code instead. That saves 2 of the about 7 per-step copies.
   - Benchmark native Linux against WSL2 if a dual boot is possible.
3. **Only then rent an H200.** The 5090 runs the `vast_h200` size already, so renting is justified only for networks too large for 32 GB, or once the WSL2 overhead is gone and compute dominates.

## Rules

- **The repo `github.com/amitb-quantum/digital_brain` is PRIVATE.** Do not change its visibility; the owner will decide when to make it public.
- **Never commit `digitalbrain.pdf`** (it's in `.gitignore`). It's the original source conversation. `docs/background.md` is its sanitised replacement. Keep that doc free of chat transcripts, chat links, usernames, hostnames and account details.
- **Secrets** (for example `VAST_API_KEY`) go in `.env`, which git ignores, or in `vastai set api-key`. Never put them in code or config.
- **vast.ai instances must be destroyed when a run finishes.** An idle H200 costs about $50 a day. Any launch script must tear the instance down.
- **Report results honestly.** Treat the claims in `docs/background.md` §4 (speedups, capacity, "perfect memory") as hypotheses to measure, not facts.
