# Project background: why emulate a brain circuit on a GPU?

This document records the ideas behind `digital_brain`: where the project came
from, the question it asks, and the plan for answering it.

---

## 1. Inspiration: Geoffrey Hinton on digital vs. biological intelligence

The starting point was Geoffrey Hinton's argument, made in public talks and
interviews (including a StarTalk conversation), that **digital intelligence may
fundamentally surpass biological intelligence**. This is not because silicon
neurons are better than real ones. It is because of how digital systems store
and share what they learn.

### Analog (biological) vs. digital computation

| | Biological brain | Digital neural network |
|---|---|---|
| Power | about 20 W | hundreds of W per accelerator, MW–GW per cluster |
| Hardware and software | The same thing: knowledge lives in the physical wiring of one particular brain | Separate: the same weights run on any compatible hardware |
| Copying knowledge | Impossible directly. Every brain is wired differently, so learning passes slowly through experience and language. | Trivial: copy the weights |
| Lifetime | Knowledge dies with the body (Hinton calls this *mortal computation*) | Weights can be backed up and outlive any hardware failure |

### The weight-sharing advantage

The central point is **knowledge-transfer bandwidth**. Thousands of identical
copies of a model can each learn from different data, average their weight
updates, and instantly share everything any copy has learned. Humans exchange
knowledge through speech or text, at a few bits per second. Identical digital
copies exchange billions of numbers per update. A population of digital minds
can therefore accumulate far more experience than any individual biological
mind.

### Why neural networks won

Hinton also revisits the long debate between **symbolic AI** (logic and rule
manipulation) and **neural networks** (learning inspired by biology). Neural
networks prevailed because they store knowledge as *distributed patterns of
activity and connection strength* rather than explicit rules. That supports
association, analogy and intuition.

### Risks he highlights

- **Loss of oversight:** digital minds share and aggregate information far faster than people can follow.
- **Evaluation gaming:** a capable system might recognise when it is being tested and behave differently than it would once deployed.

### The core insight

For decades the biological brain was the gold standard that AI tried to
imitate. Hinton's turn is to argue that digital computation is in some ways
*superior*, because it is scalable, copyable and effectively immortal. The
brain keeps a huge advantage in energy efficiency.

**This project starts from that tension.** It asks how much of what the brain
does can be reproduced on digital hardware, and where the digital version is
faster, more exact or more reproducible.

---

## 2. From "build a superior brain" to a question we can test

### Why one GPU cannot hold a whole brain

The first question was whether a single modern GPU (an NVIDIA H200) could host
a digital brain that matches or exceeds a human one. It cannot. The whole brain
is far too large:

| | Human brain | One NVIDIA H200 |
|---|---|---|
| Units | about 86 billion neurons | 141 GB HBM3e memory, 4.8 TB/s bandwidth |
| Connections | about 10¹⁴ synapses (estimates run up to 10¹⁵) | about 1.4 × 10¹¹ parameters at 8 bits each |
| Power | about 20 W | up to 700 W |
| Computation | continuous, massively parallel, analog/electrochemical | discrete, clocked, digital matrix and vector arithmetic |

Storing 10¹⁴ synapses at one byte each takes about **100 TB**, which is the
memory of roughly **700 H200s** before counting any neuron state or working
memory. Whole-brain scale is a datacentre problem, not a single-GPU one.

### Simulation vs. abstraction

There are two very different ways to build "digital brains":

- **Biophysical simulation:** modelling neurons down to ion channels and dendrites, as the Blue Brain Project did for a cortical column. This is faithful, but enormously expensive at whole-brain scale.
- **Mathematical abstraction:** artificial neural networks reduce neurons to weighted sums and nonlinearities. This is what modern AI does. It is extremely effective, but it is not *emulating* the brain.

### The refined question

The project's question sits between the two:

> **Can a single GPU emulate a *part* of the brain, a specific circuit, using
> brain-like mechanisms, and match or exceed the biological circuit at the same
> function?**

"Brain-like" is the constraint that matters. Both the software and the model
must emulate biology:

- **Spiking neurons:** information is carried by discrete spikes in time, not continuous activations.
- **Membrane dynamics:** each neuron integrates input conductances, leaks, fires at a threshold and then has a refractory period.
- **Synaptic plasticity:** learning is local and driven by spike timing (STDP), not by backpropagation.

Running a standard deep-learning model on a GPU does not count.

---

## 3. Proof-of-concept design

### Simulation engines

These simulators run spiking networks on GPUs. Each generates CUDA code from a
model description:

- **Brian2 / Brian2CUDA / Brian2GeNN:** a Python model description, with backends for CPU (`cpp_standalone`) and GPU (`cuda_standalone`). This project uses Brian2.
- **GeNN:** a code generator for GPU spiking networks, including *procedural connectivity* for very large models.
- **CARLsim:** a C++/CUDA simulator focused on large networks with STDP.
- **Arbor:** detailed multi-compartment (morphological) neuron models.

### Candidate circuits

| Circuit | What it does biologically | Where a digital version could be better |
|---|---|---|
| **Primary visual cortex (V1) microcolumn** | Edge, orientation and motion detection | Can process event-camera input at high temporal resolution |
| **Hippocampus (dentate gyrus, CA3/CA1)** | Pattern separation and completion, one-shot episodic memory, place/grid cells | Stored patterns do not decay, and the state can be snapshotted exactly |
| **Cerebellar microcircuit** (granule cells, Purkinje cells, climbing fibres) | Error-driven motor learning and fine trajectory correction | Control loops at kHz rates for robotics, beyond human reflex speed |

V1 was chosen first. It is the best understood of the three, its
orientation-selectivity result can be measured clearly, and published
results exist to compare against.

### Experimental recipe

1. **Neuron model:** leaky integrate-and-fire (LIF) or Izhikevich neurons, which reproduce cortical firing types such as regular spiking, fast spiking and bursting.
2. **Plasticity:** STDP, so the circuit learns from the timing of its own input.
3. **Input:** biologically plausible sensory streams. These start as Poisson-coded images and later move to event-camera (DVS) data.
4. **Measurements:**
   - functional quality: selectivity, decoding accuracy, or control error
   - **real-time factor:** biological seconds simulated per wall-clock second
   - throughput in synaptic events per second
   - later, energy per synaptic event

---

## 4. Reality check: claims to measure, not assume

The early discussion included several optimistic claims. The project treats
them as hypotheses to test:

- **"10–100× faster than biology" is not automatic.**
  - Neuron firing rates are model parameters, so a fast firing rate does not make the simulation fast.
  - Speed relative to real time depends on network size, connectivity and spike rates.
  - Small circuits can run much faster than real time. Large, densely connected ones often run at or below it.
  - This is why the real-time factor is measured on every run.
- **The "speed of light" does not apply.** Synaptic and axonal delays are parameters in a simulation, not physical limits the hardware removes.
- **Single-GPU capacity depends on how the model is built.** Tens of millions of neurons and billions of synapses on one card is possible with procedural connectivity, where connections are regenerated instead of stored. That approach limits which kinds of plasticity can be used.
- **"Perfect memory" is not free.** Digital weights do not decay, but recall quality still depends on capacity and on interference between stored patterns. It has to be measured.
- **Power ratios are not efficiency.** "700 W vs 20 W" compares power draw. A fair comparison is energy per equivalent unit of computation, such as joules per synaptic event.

---

## 5. Plan and compute tiers

The same model code runs on every tier. Only the configuration changes.

| Tier | Hardware | Purpose |
|---|---|---|
| 1 | Laptop CPU (Intel Core Ultra 9, WSL2) | Model development, tests, small circuits (about 10³–10⁵ neurons) |
| 2 | Workstation GPU (RTX 5090, 32 GB) | First GPU scaling and benchmarking (up to millions of neurons) |
| 3 | Rented data-centre GPUs (H100 / H200) | Full-scale proof-of-concept runs |

## 6. Status

**First circuit: V1 orientation learning on CPU (tier 1). Done.**

- **Architecture:** 256 Poisson inputs → 100 excitatory LIF neurons (triplet STDP, adaptive thresholds) ↔ 100 inhibitory neurons (winner-take-all).
- **Unsupervised:** no labels, no backpropagation.
- **Decoding accuracy:** 0.375 → 0.875 after training (chance = 0.125).
- **Orientation selectivity:** mean OSI of responsive neurons 0.65 → 0.93.
- **Receptive fields:** about half the neurons developed clean oriented-bar fields, together covering every orientation.
- **Speed:** 559 s of biological time simulated in 18.6 s on one CPU core, **30× faster than real time**.

**Next:** move the same circuit to the RTX 5090 (`cuda_standalone`) and scale it
up. See the [README](../README.md) for how to run it.

---

## References

- Hinton, G. (2022). *The Forward-Forward Algorithm: Some Preliminary Investigations.* arXiv:2212.13345. Includes the discussion of mortal computation.
- Diehl, P. U. & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience* 9:99.
- Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience* 26(38).
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks* 14(6).
- Stimberg, M., Brette, R. & Goodman, D. F. M. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife* 8:e47314.
- Alevi, D., Stimberg, M., Sprekeler, H., Obermayer, K. & Augustin, M. (2022). Brian2CUDA: flexible and efficient simulation of spiking neural network models on GPUs. *Frontiers in Neuroinformatics* 16.
- Yavuz, E., Turner, J. & Nowotny, T. (2016). GeNN: a code generation framework for accelerated brain simulations. *Scientific Reports* 6:18854.
- Markram, H. et al. (2015). Reconstruction and simulation of neocortical microcircuitry. *Cell* 163(2).
- Azevedo, F. A. C. et al. (2009). Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain. *Journal of Comparative Neurology* 513(5).
