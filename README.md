# digital_brain

**Emulating a small piece of the brain's visual cortex on a laptop CPU, a consumer gaming GPU and a data-centre GPU.**

`digital_brain` is a proof of concept for emulating brain circuits on a single GPU with biologically modelled spiking neurons, not a conventional deep-learning network:

- conductance-based membranes
- spike-timing-dependent plasticity (STDP)
- lateral inhibition

The first circuit is a model of primary visual cortex (V1). Starting from random wiring, it learns orientation-selective receptive fields, with no labels and no backpropagation.

We ran the identical model, unchanged except for its config, at three sizes and on three classes of hardware:

| Tier | Hardware |
|---|---|
| CPU | Laptop, Intel Core Ultra 9 288V, one core |
| Consumer GPU | NVIDIA RTX 5090 32 GB, Intel Core Ultra 9 285K host, WSL2 |
| Data-centre GPU | NVIDIA H100 (PCIe, SXM and NVL variants), rented by the hour on vast.ai |

The largest model has 12,800 neurons and 67 million synapses. That is roughly the synapse count of less than a tenth of a cubic millimetre of human cortex, simulated faster than real time on an H100.

> The motivation (Geoffrey Hinton on digital versus biological intelligence) and the design reasoning are in [docs/background.md](docs/background.md). The full results, with methods and caveats, are in [docs/results.md](docs/results.md).

---

## Contents

- [Headline results](#headline-results)
- [The model](#the-model)
- [Quick start](#quick-start)
- [Running experiments](#running-experiments)
- [Outputs and metrics](#outputs-and-metrics)
- [GPU setup (CUDA)](#gpu-setup-cuda)
- [Renting a data-centre GPU (vast.ai)](#renting-a-data-centre-gpu-vastai)
- [Repository layout](#repository-layout)
- [Lessons learned](#lessons-learned)
- [Limitations and next steps](#limitations-and-next-steps)
- [References](#references)

---

## Headline results

**Speed** is given as the real-time factor (RTF): biological seconds simulated per wall-clock second. Above 1× is faster than the tissue being emulated.

| Model | Synapses | Laptop CPU (1 core) | RTX 5090 (WSL2) | H100 (native Linux) |
|---|---|---|---|---|
| Small: 16×16 retina, 100 neurons | 36k | **16.4×** | 1.19× | 4.83× (PCIe) |
| Medium: 32×32 retina, 1,600 neurons | 4.2M | 0.48× | 1.15× | **4.13×** (SXM) |
| Large: 64×64 retina, 6,400 neurons | 67M | not run | 0.73× | **1.38×** (NVL) |

- **The CPU wins small networks; GPUs win large ones.** A GPU pays a fixed cost every 0.5 ms time step, about 7 tiny device-to-host copies in brian2cuda. Under WSL2 that cost is about 420 µs per step, regardless of network size. At 4.2M synapses the H100 is 8.7× faster than a laptop core.
- **At 67M synapses the GPU is compute-bound.** It is about 85% busy, so the H100's lead over the RTX 5090 shrinks from 3.6–4× to 1.9×. That figure is for the 400 W NVL card.
- **The GPU results reproduce exactly.** With the same seed, the RTX 5090 (Blackwell) and the H100 (Hopper) produced identical spike counts and learned weights.
- **Data-centre runs are cheap.** Training the large model for 2 biological hours on a rented H100 took 86 minutes and cost about USD 4.80.

**Learning:**

| Model | Mean orientation selectivity, before → after | Neurons that learned a bar | Decoding accuracy, before → after (chance 0.125) |
|---|---|---|---|
| Small | 0.57 → 0.96 | 51 / 100 | 0.49 → 0.76 |
| Medium | 0.66 → 0.997 | 594 / 1,600 | 0.89 → 0.93 |
| Large | 0.64 → 0.997 | 532 / 6,400 | 0.91 → 0.95 |

## The model

```
retina (N×N Poisson inputs) ──STDP──▶ excitatory LIF layer ──1:1──▶ inhibitory layer
                                            ▲                              │
                                            └──── lateral inhibition ◀─────┘
```

The architecture follows Diehl & Cook (2015).

| Component | Implementation |
|---|---|
| Input | One Poisson neuron per pixel, firing in proportion to a Gaussian oriented bar (250 ms on, 100 ms rest) |
| Excitatory neurons | Conductance-based leaky integrate-and-fire neurons with adaptive thresholds (θ grows with each spike during training) |
| Inhibitory neurons | One per excitatory neuron. Each inhibits every other excitatory neuron (winner-take-all). |
| Plasticity | Triplet STDP (Pfister & Gerstner 2006) on input → excitatory synapses, plus synaptic scaling that holds each neuron's total input fixed |
| Integration | Exponential Euler, time step 0.5 ms |
| Protocol | Pre-test (plasticity off), training (plasticity on), then post-test (plasticity off). All three are precomputed and simulated in a single compiled run. |

All parameters live in [config/default.yaml](config/default.yaml). The three sizes override only what differs:

| Config | Retina | Excitatory neurons | Training presentations | Peak input rate | Biological time |
|---|---|---|---|---|---|
| `config/laptop.yaml` (small) | 16×16 | 100 | 1,500 | 192 Hz | 637 s |
| `config/rtx5090.yaml` (medium) | 32×32 | 1,600 | 20,000 | 192 Hz | 7,112 s |
| `config/vast_h200.yaml` (large) | 64×64 | 6,400 | 20,000 | 96 Hz | 7,112 s |

The config names record where each size was first run. Any size runs on any backend with `--device`.

## Quick start

```bash
git clone https://github.com/amitb-quantum/digital_brain.git
cd digital_brain
conda env create -f environment.yml       # Python 3.12 + Brian2 + CUDA 12.9 toolchain
conda activate digital-brain
pytest                                    # about 3 s
python -m src.main --config config/laptop.yaml
```

On a CPU-only machine you can skip the CUDA toolchain. Any Python 3.12 environment works:

```bash
pip install -r requirements.txt
```

## Running experiments

```bash
python -m src.main --config config/laptop.yaml                            # small model, CPU (about 40 s)
python -m src.main --config config/rtx5090.yaml                           # medium model, CUDA GPU
python -m src.main --config config/vast_h200.yaml                         # large model, CUDA GPU (needs about 6 GB host RAM)
python -m src.main --config config/rtx5090.yaml --device cpp_standalone   # any size on the CPU
python -m src.main --config config/vast_h200.yaml --n-presentations 200 --no-plots   # short speed check
```

| Option | Meaning |
|---|---|
| `--config` | Tier file, deep-merged over `config/default.yaml` |
| `--device` | `runtime` (numpy), `cpp_standalone` (compiled C++) or `cuda_standalone` (brian2cuda) |
| `--n-presentations` | Override the number of training presentations |
| `--out` | Output root (default `data/runs/`) |
| `--keep-build` | Keep the generated C++/CUDA project |
| `--no-plots` | Skip the receptive-field and selectivity figures |

Approximate wall time for each model:

| Model | Laptop CPU | RTX 5090 (WSL2) | H100 |
|---|---|---|---|
| Small | 40 s | 9 min | 2 min |
| Medium | 4.2 h | 1.7 h | 29 min |
| Large | about 17 h (estimated, not run) | 2.7 h | 86 min |

## Outputs and metrics

Each run writes `data/runs/<config>-<timestamp>/`:

| File | Contents |
|---|---|
| `metrics.json` | Every number reported above |
| `config.yaml` | The fully merged config used |
| `results.npz` | Initial and final weights, final thresholds, excitatory spike times |
| `rf_initial.png`, `rf_final.png` | Every neuron's input weights drawn as an image |
| `selectivity.png` | OSI histograms, preferred orientations and tuning curves |

Metrics:

- **Orientation selectivity index (OSI):** circular variance of each neuron's tuning curve, from 0 (untuned) to 1 (responds to one orientation only). Only neurons with at least `test.min_spikes` test spikes count, because OSI is inflated for neurons that barely fire.
- **Learned receptive fields (`learned_rfs`):** neurons whose weights correlate with an oriented-bar template at r > 0.5. The distribution is bimodal, and learned neurons sit above 0.8. This metric counts neurons that winner-take-all keeps silent during the test.
- **Decoding accuracy:** label each neuron by its preferred orientation on test repeat 0, then classify later repeats by label-group response, following Diehl & Cook. Chance is 1/8. At large sizes random weights already decode well, so OSI and learned receptive fields are the better learning measures.
- **Real-time factor and synaptic events per second:** simulation speed, excluding compilation.

## GPU setup (CUDA)

The GPU tiers use [brian2cuda](https://github.com/brian-team/brian2cuda), which generates and compiles a standalone CUDA project. `environment.yml` installs everything from conda-forge without root:

- the CUDA 12.9 toolkit (`cuda-nvcc`, `cuda-cudart-dev`, `libcurand-dev`, `cuda-profiler-api`)
- conda's gcc 14, which CUDA 12.9 supports (gcc 15 is not)

Things that are not obvious:

- **Versions are coupled.** brian2cuda 1.0b1 pins Brian2 2.10.1 exactly, and Brian2 2.10 needs Python 3.12 or newer.
- **Use CUDA 12.x, not 13.** brian2cuda compiles with `-std=c++11`, and CUDA 13's CCCL needs C++17.
- **gcc 14's headers do not parse as C++11.** `configure_device()` appends `-std=c++17`, and nvcc keeps the last one. Its "incompatible redefinition" warning is expected.
- **Blackwell works.** The RTX 5090 is compute capability 12.0 (`sm_120`) and needs CUDA 12.8 or newer.
- **On WSL2, the NVIDIA driver comes from Windows.** Do not install a Linux driver inside WSL. Expect a per-step overhead of about 420 µs (see the results above).

## Renting a data-centre GPU (vast.ai)

[scripts/vast_run.py](scripts/vast_run.py) runs one experiment on a rented GPU under a hard cost cap:

```bash
pip install vastai && vastai set api-key <your key>          # key is stored in ~/.config/vastai
python scripts/vast_run.py quote --gpu H100_SXM,H100_NVL --country US --max-cost 5    # read-only
python scripts/vast_run.py launch --gpu H100_SXM --country US --max-hourly 2.50 \
       --config config/rtx5090.yaml --max-cost 5            # dry run: prints the plan
python scripts/vast_run.py launch ... --yes                  # rents, runs, collects, destroys
```

Safety design:

- **Hard cost cap.** The rental lifetime is capped at `(max_cost − 0.50) × 3600 / max_hourly` seconds.
- **Independent watchdog.** It is armed before the instance is created and destroys the instance at the deadline even if the launcher crashes.
- **Calibration before the full run.** A 200-presentation run on the rented host measures speed and forecasts the full run. If the forecast does not fit under the cap, the launcher aborts within minutes. This matters because Brian2 standalone writes results only at the end.
- **Destroy, then verify.** Instances are destroyed by matching label and ID, then confirmed absent from the account.
- **Code is pinned to a commit.** The payload is `git archive HEAD`, so the launcher refuses to run with uncommitted changes.
- **Remote setup is reproducible.** [scripts/vast_remote_setup.sh](scripts/vast_remote_setup.sh) installs a pinned Miniforge (checksum verified) and the same `environment.yml`, then runs the test suite.

## Repository layout

```
config/            default.yaml plus one override file per model size
docs/
  background.md    motivation and design reasoning
  results.md       full results, methods, lessons and limitations
scripts/
  vast_run.py      cost-capped vast.ai launcher (quote / launch / cleanup)
  vast_remote_setup.sh
src/
  network.py       Brian2 model and backend selection
  stimulus.py      oriented bars and the precomputed run schedule
  metrics.py       OSI, decoding, learned receptive fields, throughput
  main.py          command-line entry point
  plots.py         receptive-field and selectivity figures
  config.py        config loading and deep merge
tests/             pytest suite (smoke run, metrics, stimulus, launcher logic)
environment.yml    full environment, including the CUDA toolchain
requirements.txt   Python packages only
```

## Lessons learned

1. **Forward Euler can blow up a spiking network at scale.** At dt 0.5 ms, a 1,600-neuron winner-take-all layer delivered enough simultaneous inhibition to push Euler past the inhibitory reversal potential. Every neuron then fired at about 180 Hz. Exponential Euler fixes this at the same step size.
2. **Scale input drive with retina size.** Bars get longer as the retina grows. At 64×64, the 192 Hz rate that suits 32×32 left 71% of neurons firing before any learning, and 96 Hz restored sparse activity.
3. **Copy Brian2 monitor arrays.** In Brian2 2.10, `np.asarray(mon.i[:])` returns a view into memory that is freed along with the network. Use `np.array`.
4. **GPU speedups depend on network size.** Measure at the size you care about. A 3.6× advantage at 4M synapses became 1.9× at 67M.
5. **Calibrate before committing a rental.** A short run on the rented host predicts the full run's speed to within about 10–20%.

More detail is in [docs/results.md](docs/results.md#lessons-we-would-pass-on).

## Limitations and next steps

- **Recruitment:** at the larger sizes only 8–37% of neurons learn a receptive field. Recruiting more is the main open problem.
- **Seeds:** the medium and large GPU results each come from one seed. They reproduce bit-for-bit across GPU architectures, but not across seeds.
- **WSL2:** the RTX 5090 was measured under WSL2. Native Linux would likely narrow its gap to the H100 at small and medium sizes.
- **Scope:** this is one simplified V1-like circuit learning one stimulus class. It is a hardware and methods benchmark, not a claim about brain-scale intelligence.
- **Next circuits** (see [docs/background.md](docs/background.md)): hippocampal pattern completion (CA3/CA1) and a cerebellar microcircuit for motor control.

## References

- Diehl, P. U. & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience* 9:99.
- Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience* 26(38).
- Stimberg, M., Brette, R. & Goodman, D. F. M. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife* 8:e47314.
- Alevi, D., Stimberg, M., Sprekeler, H., Obermayer, K. & Augustin, M. (2022). Brian2CUDA: flexible and efficient simulation of spiking neural network models on GPUs. *Frontiers in Neuroinformatics* 16:883700.
