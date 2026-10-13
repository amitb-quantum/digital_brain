# Results: one spiking V1 circuit on a laptop CPU, a consumer GPU and a data-centre GPU

*October 2026. Code: this repository, branch `rtx5090-bringup`.*

## Summary

We ran a biophysical spiking model of primary visual cortex (V1) on three classes of hardware, using identical code at three network sizes. The model has conductance-based LIF neurons, triplet STDP, adaptive thresholds and winner-take-all inhibition, after Diehl & Cook (2015). It learns orientation-selective receptive fields from oriented bars, with no labels and no backprop.

- **The circuit learns at every size and on every tier.** Mean orientation selectivity (OSI) rises from about 0.65 to 0.96–0.997, and neurons develop clean oriented-bar receptive fields.
- **The best hardware depends on network size.** At 36k synapses a single laptop CPU core is fastest. At 4.2M synapses an H100 runs 8.7× faster than the laptop. At 67M synapses the GPU is compute-bound and the H100's lead over the RTX 5090 narrows to 1.9×.
- **The GPU results are reproducible to the bit.** The same seed gives identical spike counts and learned weights on an RTX 5090 (Blackwell) and an H100 (Hopper).
- **Data-centre runs are cheap.** The largest model trained on a rented H100 in 86 minutes for about USD 5.

## Model and protocol

| | Small | Medium | Large |
|---|---|---|---|
| Retina (Poisson inputs) | 16×16 | 32×32 | 64×64 |
| Excitatory / inhibitory neurons | 100 / 100 | 1,600 / 1,600 | 6,400 / 6,400 |
| Synapses | 36k | 4.2M | 67M |
| Training presentations | 1,500 | 20,000 | 20,000 |
| Peak input rate | 192 Hz | 192 Hz | 96 Hz |
| Biological time simulated | 637 s | 7,112 s | 7,112 s |

- **Stimulus:** each presentation is a Gaussian bar at a random orientation and offset, shown for 250 ms, followed by 100 ms of rest.
- **Test blocks:** identical pre-training and post-training blocks with plasticity off, 8 orientations × 5 offsets × 4 repeats each.
- **Integration:** time step 0.5 ms, exponential Euler.
- **Simulator:** Brian2 2.10.1 in standalone mode. The CPU uses `cpp_standalone` on 1 thread; the GPUs use brian2cuda 1.0b1 with CUDA 12.9.
- **Single compiled run:** the whole pre-test, training and post-test timeline is precomputed and executed as one compiled run.

**Real-time factor (RTF)** is biological seconds simulated per wall-clock second. It excludes compilation, which takes 4–41 s.

## Speed

| Model | Laptop CPU (Core Ultra 9 288V, 1 thread) | RTX 5090 32 GB (WSL2) | H100 (rented, native Linux) |
|---|---|---|---|
| Small (36k synapses) | **RTF 16.4** | RTF 1.19 | RTF 4.83 (H100 PCIe) † |
| Medium (4.2M synapses) | RTF 0.48 | RTF 1.15 | **RTF 4.13** (H100 SXM) |
| Large (67M synapses) | not run (about 6 h estimated) | RTF 0.73 | **RTF 1.38** (H100 NVL); SXM: *pending* |

Throughput in synaptic events per wall-clock second:

| Model | Laptop CPU | RTX 5090 | H100 |
|---|---|---|---|
| Medium | 8.4×10⁶ | 2.0×10⁷ | 7.3×10⁷ |
| Large | – | 5.2×10⁷ | 9.9×10⁷ |

† Measured with an earlier, shorter test schedule (559 s of biological time). At this size every GPU is latency-bound, so the per-step cost and the RTF are essentially unchanged. The RTX 5090 gave RTF 1.24 on the old schedule and 1.19 on the current one.

### Why the curves cross

- **Per-step overhead dominates small networks.** brian2cuda makes several small device-to-host copies every time step, about 7 for this model, to size kernel launches and count spikes.
  - **Under WSL2**, we measured 7 µs per kernel launch, 23 µs per sync and 77 µs per 4-byte copy. That gives the 5090 a floor of about 400 µs per 0.5 ms step, whatever the network size: the small and medium models cost the same per step.
  - **On native Linux**, the H100 needed about 100 µs per step on the small model.
  - **A single CPU thread** has no such overhead and wins outright at 36k synapses.
- **Compute dominates large networks.** At 67M synapses the GPU is about 85% busy, so the H100's advantage over the 5090 falls from about 3.6–3.9× at the small and medium sizes to 1.9×, using the 400 W H100 NVL.
- **Caveat:** the 5090 figures include the WSL2 overhead. A native-Linux 5090 would likely close much of the gap at the small and medium sizes. We have not measured this.

## Learning

| | Small (laptop CPU) | Medium (H100 SXM) | Large (H100 NVL) |
|---|---|---|---|
| Mean OSI, pre → post | 0.57 → 0.96 | 0.66 → 0.997 | 0.64 → 0.997 |
| Learned receptive fields | 51 / 100 | 594 / 1,600 (37%) | 532 / 6,400 (8.3%) |
| Decoding accuracy, pre → post (chance 0.125) | 0.49 → 0.76 | 0.89 → 0.93 | 0.91 → 0.95 |

- **Learned receptive fields** are neurons whose input weights correlate with an oriented-bar template at r > 0.5. The distribution is bimodal: almost every such neuron has r > 0.8.
- **The small model is stable across seeds:** over seeds 42, 1, 2 and 3 on a CPU, it learned 47–51 receptive fields and decoding rose from 0.49 to 0.80 on average.
- **Decoding is a weak learning signal at scale.** With thousands of neurons, random weights already decode orientation at about 0.9. We therefore report learned receptive fields and OSI as the main learning metrics.
- **Recruitment is the open problem.** Learned neurons accumulate at about one per 20–40 training presentations. 20k presentations recruit 37% of 1,600 neurons but only 8% of 6,400.

## Reproducibility

- **GPU against GPU: bit-identical.** The medium and large models produced identical spike counts and learned weights on the RTX 5090 and on the H100: 776,326 and 945,729 excitatory spikes respectively.
- **CPU against CPU: identical across machines.** The laptop and a desktop CPU produced identical small-model results.
- **CPU against GPU: close, not identical.** The random-number streams differ, so results are close but not bit-identical (medium model: 587 against 594 learned receptive fields).

## Lessons we would pass on

1. **Forward Euler can blow up a spiking network at scale.**
   - At dt 0.5 ms, a large winner-take-all layer can deliver enough simultaneous inhibition (gi above about 400) that Euler overshoots past the inhibitory reversal potential. Every neuron then fires at about 180 Hz, on CPU and GPU alike.
   - Exponential Euler is exact for the membrane equation given the conductances, and it fixes this at the same time step.
   - The parameters were hand-tuned under Euler, so switching integrators cost the small model about 0.1 decoding accuracy.
2. **Scale input drive with retina size.** Bars get longer as the retina grows. Keeping 192 Hz at 64×64 made 71% of neurons fire before any learning. Halving the rate to 96 Hz restored sparse, learnable activity (15%).
3. **brian2cuda on current toolchains.**
   - brian2cuda 1.0b1 pins Brian2 2.10.1 exactly, which needs Python 3.12 or newer.
   - Use CUDA 12.x rather than 13: brian2cuda builds with `-std=c++11`, and CUDA 13's CCCL needs C++17.
   - gcc 14's libstdc++ headers do not parse as C++11 under nvcc. Append `-std=c++17` (nvcc keeps the last one).
   - conda-forge needs `cuda-profiler-api` alongside `cuda-nvcc`.
   - Blackwell (sm_120) works.
4. **Brian2 2.10 monitor arrays are views into memory owned by the network.** Copy them with `np.array(mon.i[:])`, not `np.asarray`, or you read freed memory once the network is gone.
5. **Calibrate before you commit a rental.**
   - Standalone Brian2 writes results only when the run ends, so a run cut off by a budget cap returns nothing.
   - A 3-minute calibration of the same config on the rented host predicts the full run well. It under-predicts by 9–22%, because short runs are dominated by the plasticity-free test blocks.
   - We lost one run (about USD 4) before adding this check. Afterwards it aborted an under-budgeted run for about USD 0.30.
6. **Cloud logistics.**
   - vast.ai re-issues offer IDs for the same machine between searches, so approve a machine and a maximum price rather than an offer ID.
   - A US host set up in 38 s, against 1.5–4.7 min for hosts overseas.
   - One host never became reachable; a 20-min SSH timeout plus automatic destruction limited the loss to USD 0.57.

## Cost

| Run | GPU | Time | Cost (USD) |
|---|---|---|---|
| Small model | H100 PCIe | 6 min | 0.19 |
| Medium model | H100 SXM | 36 min | 1.16 |
| Large model | H100 NVL | 95 min | 4.80 |
| Overheads: a crashed launcher, a cut-off run, a dead host, a calibration abort | – | – | about 5.25 |
| Large model | H100 SXM | *pending* | *pending* |

## Reproduce

```bash
conda env create -f environment.yml && conda activate digital-brain
python -m src.main --config config/laptop.yaml                            # small, CPU
python -m src.main --config config/rtx5090.yaml                           # medium, CUDA
python -m src.main --config config/vast_h200.yaml                         # large, CUDA
python -m src.main --config config/rtx5090.yaml --device cpp_standalone   # any size on CPU
python scripts/vast_run.py quote --gpu H100_SXM --country US              # rented H100 (read-only quote)
```

Each run writes `metrics.json` (with OSI, decoding, learned receptive fields, RTF and synaptic events per second), `results.npz` and receptive-field plots to `data/runs/<config>-<timestamp>/`.

## Limitations

- **Seeds:** the medium and large GPU results each come from a single seed. They were reproduced bit-for-bit across two GPU architectures, but not across seeds.
- **WSL2:** all RTX 5090 figures were measured under WSL2.
- **Mixed H100 variants:** the H100 column mixes PCIe (small), SXM (medium) and NVL (large) parts.
- **Recruitment:** only 8–37% of neurons learn a receptive field at the larger sizes.
- **Simplified biology:** this is one V1-like circuit learning one stimulus class. It is a hardware and methods benchmark, not evidence of brain-scale capability.
