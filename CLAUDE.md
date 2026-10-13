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

- **Conda env `digital-brain`, Python 3.12.** Build it with `conda env create -f environment.yml`, which includes the CUDA toolkit. On a CPU-only machine, `pip install -r requirements.txt` on Python 3.12 is enough.
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

### The three-tier showcase

The PoC compares three classes of hardware on the same model:

1. **CPU:** the Dell laptop.
2. **Consumer GPU:** the RTX 5090 in vermithor.
3. **Data-centre GPU:** an H100 or H200 rented on vast.ai.

Keep all three. Vermithor's CPU is useful for debugging but does not stand in for tier 1.

For a fair comparison, run each model size on each tier, with the same code, integrator and seed. `--device` runs any size config on any backend:

```bash
python -m src.main --config config/rtx5090.yaml --device cpp_standalone   # medium model on the CPU
```

| Model size (config) | Synapses | Laptop CPU | RTX 5090 | H100/H200 |
|---|---|---|---|---|
| Small: 16×16 retina, 100 exc (`laptop.yaml`) | 36k | to re-measure (RTF about 30 under `euler`) | RTF 1.24 | **RTF 4.83** (H100 PCIe) |
| Medium: 32×32 retina, 1600 exc (`rtx5090.yaml`, 20k presentations) | 4.2M | not run | RTF 1.15 | **RTF 4.13** (H100 SXM) |
| Large: 64×64 retina, 6400 exc (`vast_h200.yaml`) | 67M | not run (too slow; estimate from a short run) | RTF 0.47 (short run) | not run |

- **Re-run the laptop column under `exponential_euler`.** The laptop's last numbers used `euler`.
- **The laptop env needs the new stack first:** Python 3.12 and `pip install -r requirements.txt`. The laptop does not need the CUDA toolkit.
- **Fill in the matrix only after retuning** (see Next steps). Otherwise it benchmarks a network that does not learn at the medium and large sizes.

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

**Current small-model reference** (`laptop.yaml` with `exponential_euler`, run on vermithor's CPU, seeds 42, 1, 2, 3):
- The test set is now 5 offsets × 4 repeats: 120 decoded trials, up from 24.
- Decoding accuracy 0.49 → 0.80 (mean; per seed 0.76 to 0.83).
- Mean OSI 0.67 → 0.96.
- Learned RFs (bar-template r > 0.5, now in `metrics.json` as `learned_rfs`): 0 → 49 of 100 (range 47 to 51).
- RTF about 30.

On the larger test, random weights decode at about 0.49 rather than 0.375, so the old 24-trial numbers overstated the before/after gap.

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

### First vast.ai run (2026-10-11, H100 PCIe 80 GB, machine 149333, $2.40/h)

- **Run:** `rtx5090_validate.yaml` (the small model) on `cuda_standalone` under native Linux, driver 595.71.
  - RTF 4.83: 558.6 s of biological time in 115.6 s, plus 15.6 s compile.
  - About 103 µs per step, against about 400 on the 5090 under WSL2, so 3.9× faster at this size.
  - This supports the WSL2 sync-latency explanation for the 5090's per-step floor.
- **Learning** matched the CPU run under `exponential_euler`: accuracy 0.33 → 0.71 and OSI 0.68 → 0.93.
- **Timeline and cost:** about 6 minutes from create to destroy (SSH 81 s, setup about 85 s, run 132 s), and about $0.19 of credit.
- **Launcher bugs this rental exposed:**
  1. `vastai attach ssh` prints a Python repr, which crashed the first attempt. Cleanup destroyed that instance within 4 s.
  2. `cd X && cmd &` held ssh open until the job ended, which would have killed long runs at the 600 s ssh timeout. Both are fixed.

### Medium model on an H100 SXM (2026-10-11, machine 150369, $2.08/h)

- **Run:** `rtx5090.yaml` with 20k presentations: 7112 s of biological time in 1724 s, plus 16 s compile.
  - RTF 4.13, against 1.15 on the 5090 under WSL2: 3.6× faster.
  - 7.3×10⁷ synaptic events per second.
- **Bit-identical to the 5090 run:** exactly 776,326 excitatory spikes, 594 learned RFs, OSI 0.659 → 0.997 and accuracy 0.892 → 0.933. brian2cuda with a fixed seed is deterministic across Blackwell and Hopper, so the GPU tiers differ only in speed.
- **Rental:** about 36 min (SSH 111 s, setup 4 min 41 s, run 29 min), about $1.14 of credit.

### Recruitment at the medium size (32×32 retina, 1600 exc)

A "learned" neuron has a best Pearson r above 0.5 against oriented-bar templates. The split is sharp: almost all of these have r above 0.8, and the rest show nothing.

| Training presentations | Learned neurons | Presentations per learned neuron |
|---|---|---|
| 1,500 | 75 (4.7%) | 20 |
| 2,500 | 110 (6.9%) | 23 |
| 5,000 | 209 (13.1%) | 24 |
| 20,000 | 594 (37.1%) | 34 |
| 1,500 at laptop size (100 exc), for comparison | 48 (48%) | 31 |

- **Recruitment is near-linear but slows with training:** about 20 presentations per neuron early on, about 34 by 20k. The medium model was under-trained, not broken.
- **The 20k run** (5090, RTF 1.15, 1.7 h, 120-trial test):
  - learned RFs 0 → 594
  - mean OSI 0.66 → 0.997
  - decoding 0.89 → 0.93
  - responsive fraction 0.24 → 0.10
- **`rtx5090.yaml` now trains for 20,000 presentations.**
- **At this size, decoding is a weak learning signal.** With 1600 neurons, random weights already decode orientation at 0.89. Report learned RFs and OSI as the main learning metrics.
- **Post-test responsive fraction is a poor scale metric.** It uses 24 distinct test stimuli, and winner-take-all lets only a few neurons win each one.
- **Decoding accuracy is just as unreliable here.** It uses 24 evaluation trials, and random weights already decode at 0.75 to 0.79 at this size. Fix the evaluation before tuning against it.

### Large model input drive (2026-10-12, 5090, 500 training presentations)

| `max_rate_hz` | Pre-test responsive | Spikes per presentation | Learned RFs | Accuracy | RTF |
|---|---|---|---|---|---|
| 192 (old) | 0.71 | 2283 | n/a | n/a | 0.47 |
| **96 (now in `vast_h200.yaml`)** | 0.15 | 221 | 30 | 0.82 → 0.99 | 0.67 |
| 128 | 0.30 | 370 | 37 | 0.79 → 0.92 | 0.66 |

- **Why 96:** bars are twice as long at 64×64, so halving the rate keeps input per presentation about the same as the working medium model.
- **Recruitment** is about one neuron per 15–20 presentations, so 20k presentations should recruit about 1000 of the 6400.

## Next steps

1. **Retune for scale.** Make the learning work at 32×32 with 1600 neurons before running anything bigger.
   - Strengthen the evaluation first: more test offsets and repeats, a learned-RF count from the bar-template score (now in a scratch script), and held-out positions.
   - Then decide between longer training (linear recruitment) and faster recruitment per presentation.
   - Scale input drive with retina size, for example `max_rate_hz` or `norm_mean_w`, so pre-test activity stays sparse.
   - Recruit more neurons: longer training, or a larger `theta_plus` with fewer neurons.
   - Retune laptop size under `exponential_euler` to recover the about 0.1 decoding gap.
2. **Cut per-step GPU overhead.**
   - Drop the input and inhibitory `SpikeMonitor`s, which are used only for counts, and count spikes with a variable in the reset code instead. That saves 2 of the about 7 per-step copies.
   - Benchmark native Linux against WSL2 if a dual boot is possible.
3. **Only then rent an H200.** The 5090 runs the `vast_h200` size already, so renting is justified only for networks too large for 32 GB, or once the WSL2 overhead is gone and compute dominates.

## vast.ai tier: `scripts/vast_run.py`

```bash
python scripts/vast_run.py quote --max-cost 3        # read-only: credit, instances, offers, lifetime caps
python scripts/vast_run.py launch --machine-id M --max-hourly 2.5 --config config/rtx5090.yaml --max-cost 3   # dry run
python scripts/vast_run.py launch ... --yes          # rents; only after the owner approves the quote
python scripts/vast_run.py cleanup --campaign data/vast/LABEL   # manual destroy + verify
```

- **Lifetime cap.** The lifetime is `floor((max_cost - 0.50) * 3600 / hourly)`, where hourly includes 40 GB of disk.
- **Watchdog.** An independent watchdog is armed before the create call. At the deadline it destroys every instance carrying the campaign's unique label. Keep vermithor awake and WSL running until the campaign log shows the instance destroyed.
- **Refusals.** The launcher will not start if the git tree is dirty (the payload is `git archive HEAD`), if any instance already exists on the account (it is shared), or if the credit does not cover the cap.
- **Remote setup.** `scripts/vast_remote_setup.sh` installs pinned Miniforge 26.5.3-0 (checksum verified), creates the env from `environment.yml`, then runs `pip check` and `pytest`. It has been validated locally from scratch. The image is pinned by digest: `vastai/base-image:cuda-12.9-mini-py312-2026-09-07`.
- **Campaign folders.** `data/vast/<label>/` holds the state, logs, the per-rental private SSH key and the collected results. Git ignores it.
- **Design.** The design follows the SpectraNS-3D vast.ai handoff: search is not rental, destroy by label and ID then verify absence, check for the label before retrying a lost create, and collect small files first. Do not use that project's scripts, key or budget.

## Rules

- **The repo `github.com/amitb-quantum/digital_brain` is PRIVATE.** Do not change its visibility; the owner will decide when to make it public.
- **Never commit `digitalbrain.pdf`** (it's in `.gitignore`). It's the original source conversation. `docs/background.md` is its sanitised replacement. Keep that doc free of chat transcripts, chat links, usernames, hostnames and account details.
- **Secrets** (for example `VAST_API_KEY`) go in `.env`, which git ignores, or in `vastai set api-key`. Never put them in code or config.
- **vast.ai budget: USD 10 ceiling for this project** (set 2026-10-10; the owner may revise it).
  - Every rental needs the owner's approval of a fresh quote.
  - The account is shared with another project. Before renting, check the live balance and that no other instance is running. Never spend another project's allowance.
- **vast.ai instances must be destroyed when a run finishes.** An idle H200 costs about $50 a day. Any launch script must tear the instance down.
- **Report results honestly.** Treat the claims in `docs/background.md` §4 (speedups, capacity, "perfect memory") as hypotheses to measure, not facts.
