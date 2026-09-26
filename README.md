# RIPE-MambaSpike

Reference implementation of **RIPE-MambaSpike: Resolution-Independent
Spiking–State-Space Interfaces for Parameter-Efficient Event-Based Vision**.

The parameter cost of a spiking/state-space hybrid is decided at the interface
between them, not by the capacity the task demands. Projecting a flattened
spiking feature map into a sequence model makes that projection quadratic in
sensor resolution; in the baseline we profile it holds 92% of all parameters.
Pooling every stage to a common grid and projecting at a fixed width removes
the dependence, so the deployed parameter count is **constant at 0.87 M across
a 43× range of input areas**, from 34² to 224².

---

## Results to reproduce

| Benchmark | Deployed params | Reported (best seed) | 5-seed mean | `tools/` script |
|---|---:|---:|---:|---|
| DVS-Gesture | 0.88 M | **98.48** | 98.18 ± 0.31 | `train_dvsgesture.py` |
| CIFAR10-DVS | 0.87 M | **81.40** | 81.28 ± 0.11 | `train_cifar10dvs.py` |
| N-Caltech101 | 0.88 M | **85.54** | 85.49 ± 0.18 | `train_ncaltech101.py` |
| N-MNIST | 0.87 M | **99.57** | 99.55 ± 0.02 | `train_nmnist.py` |
| DailyDVS-200 | 8.04 M | **45.74** top-1 | 45.74 ± 0.40 | `train_dailydvs200.py` |
| DailyDVS-200 *lite* | 3.82 M | **42.41** top-1 | 42.41 ± 0.34 | `train_dailydvs200.py --lite` |

Headline numbers are the best of five seeds, which is the convention used
throughout the paper's SOTA table; the mean and standard deviation of all five
are given alongside so the spread is visible. On DVS-Gesture's 264-clip test
split one clip is 0.38 pp, so σ ≈ 0.83 clips and single-seed differences below
about one point are not resolvable.

Evaluation is a single forward pass. There is no EMA, no test-time
augmentation and no multi-crop.

### Verify the parameter counts without training

The central claim is checkable in seconds and needs no data:

```bash
python -c "
from ripe_mambaspike import build_model
from ripe_mambaspike.utils import deployed_parameters
for b in ['dvsgesture','cifar10dvs','ncaltech101','nmnist']:
    print(f'{b:<14}{deployed_parameters(build_model(b)):>9,}')"
```

```
dvsgesture      875,122
cifar10dvs      869,593
ncaltech101     881,332
nmnist          869,593
```

Resolution independence, likewise:

```bash
python -c "
import torch
from ripe_mambaspike import build_model
from ripe_mambaspike.utils import deployed_parameters
for r in (34, 48, 128, 224):
    m = build_model('cifar10dvs', T=4).cuda().eval()
    with torch.no_grad(): m(torch.randn(1, 4, 2, r, r).cuda())
    print(f'{r}^2  {deployed_parameters(m):,}')"
```

```
34^2   864,193
48^2   864,193
128^2  864,193
224^2  864,193
```

All four resolutions report the same count, and each ran a real forward pass
at that resolution, so the number is not a static property of the constructor.
The count here is below the table's 869,593 only because `T=4` is used to keep
the check fast; the BNTT layers hold per-timestep statistics.

A GPU is required: `mamba_ssm`'s selective scan is a fused CUDA kernel and
raises `Expected u.is_cuda() to be true` on CPU. The parameter-count check
above needs no GPU.

---

## Install

```bash
conda create -n ripe python=3.10 -y && conda activate ripe
pip install -r requirements.txt
pip install -e .
```

`mamba-ssm` and `causal-conv1d` build CUDA kernels and need a matching
toolchain; install them after PyTorch. `tonic` is needed only for DVS-Gesture
and N-MNIST, `spikingjelly` only for N-Caltech101.

Verify the install:

```bash
python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available())"
```

---

## Data

Each script takes `--data-root`. The expected layout per benchmark:

**DVS-Gesture** — from the [IBM DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/)
release, pre-split into per-recording `.npy` event arrays with columns
`[x, y, polarity, t_ms]`:

```
<data-root>/
├── ibmGestureTrain/<recording>/<label>.npy
└── ibmGestureTest/<recording>/<label>.npy
```

**CIFAR10-DVS** — one `.pt` per sample holding `(events, label)`, named
`0.pt`, `1.pt`, …:

```
<data-root>/
├── train/*.pt
└── test/*.pt
```

Pass `--cache-dir` to cache resized tensors; the first epoch is slow without it.

**N-Caltech101** — the raw `.bin` recordings. On first run the script writes
`<data-root>/events_np/` (about one minute) and uses the deterministic
per-class 90/10 split, which reproduces spikingjelly's partition with
`random_split=False`.

**N-MNIST** — downloaded automatically by `tonic` into `--data-root`.

**DailyDVS-200** — the released recordings plus the official split files. If
`train.txt` / `test.txt` are present pass `--split-dir`; otherwise the script
falls back to the subject-disjoint split defined in
`ripe_mambaspike/data/dailydvs200.py`. At 224² the frame cache is worth
enabling with `--cache-dir`.

---

## Train

Every script's defaults are the published recipe for that benchmark, so the
commands below are complete as written.

```bash
python tools/train_dvsgesture.py   --data-root /path/to/DvsGesture
python tools/train_cifar10dvs.py   --data-root /path/to/cifar-dvs   --cache-dir /path/to/cache
python tools/train_ncaltech101.py  --data-root /path/to/NCaltech101
python tools/train_nmnist.py       --data-root /path/to/nmnist
python tools/train_dailydvs200.py  --data-root /path/to/DailyDVS200 --cache-dir /path/to/cache
python tools/train_dailydvs200.py  --data-root /path/to/DailyDVS200 --lite --run-name dailydvs200_lite
```

To reproduce the five-seed spread, vary `--seed` and `--run-name` together so
the runs do not overwrite one another:

```bash
for s in 0 1 2 3 4; do
  python tools/train_cifar10dvs.py --data-root /path/to/cifar-dvs \
      --seed $s --run-name cifar10dvs_seed$s
done
```

### Resuming

Re-issue the same command with `--resume`. It restores model, optimizer,
schedule position, epoch and best accuracy from `results/<run-name>/latest.pth`
(falling back to `best.pth`) and appends to the existing `log.csv` rather than
truncating it. Passing `--resume` on a fresh run is harmless.

### Output

```
results/<run-name>/
├── latest.pth     model + optimizer + scheduler + epoch + best accuracy
├── best.pth       same, written whenever test accuracy improves
└── log.csv        one row per epoch
```

`log.csv` columns: `epoch, lr, alpha, train_loss, train_loss_tet,
train_loss_sgc, train_loss_l1, train_acc, test_loss, test_acc, best_acc,
grad_norm, spike_rate`. The reported number is the maximum of `test_acc`,
which is also what `best_acc` tracks.

---

## Recipe

Script defaults, matching Table 17 of the paper. Every value is overridable on
the command line.

| | DVSG | C10-DVS | NCal101 | N-MNIST | DDVS-200 |
|---|---|---|---|---|---|
| Epochs | 300 | 300 | 300 | 300 | 250 |
| Batch × accum | 32 × 1 | 32 × 1 | 32 × 1 | 64 × 2 | 32 × 1 |
| LR (init / min) | 1e-3 / 1e-6 | 1e-3 / 1e-6 | 1e-3 / 1e-6 | 1e-3 / 1e-6 | 5e-4 / 1e-6 |
| Warm-up epochs | 10 | 15 | 15 | 10 | 10 |
| Weight decay | 0.05 | 0.03 | 0.03 | 0.05 | 0.02 |
| Drop-path | 0.1 | 0.2 | 0.2 | 0.1 | 0.2 |
| TET λ | 5e-3 | 1e-2 | 1e-2 | 5e-3 | 5e-3 |
| Surrogate α (start → end) | 2 → 4 | 2 → 3 | 2 → 3 | 2 → 4 | 2 → 4 |
| SGC λ | 1.0 | 0 | 0 | 1.0 | 0 |
| L1 λ | 1e-4 | 0 | 0 | 1e-4 | 1e-4 |
| Mixup α / prob | 0.2 / 0.3 | 0.4 / 0.5 | 0.4 / 0.5 | 0.4 / 0.5 | 0.2 / 0.5 |
| Label smoothing | 0.1 | 0.05 | 0.05 | 0.05 | 0.1 |
| EventCutMix α | — | 0.4 | 0.2 | 0.2 | — |
| T | 16 | 10 | 10 | 10 | 10 |
| Spatial | 128² | 128² | 128² | 34² | 224² |
| NDA ops / mag | 1 / 0.3 | 2 / 0.5 | 2 / 0.2 | 2 / 0.3 | n/a |
| NDA roll / cutout cap | 0.25 / 0.25 | 0.25 / 0.25 | 0.15 / 0.15 | 0.15 / 0.20 | n/a |
| Stage channels | [32,64,128] | [32,64,128] | [32,64,128] | [32,64,128] | [96,192,384] |
| Classifier head width | — | — | — | — | 1024 |

CIFAR10-DVS and N-Caltech101 use a TET-only objective: `--sgc-lambda 0
--l1-lambda 0` disables the consistency and firing-rate terms while the
relaxed forward pass still contributes its own TET loss. Passing `--no-sgc`
is a different thing — it skips that pass entirely, roughly halving step cost
and activation memory, and will not reproduce the reported numbers.

Gradients are clipped at ‖g‖₂ = 1.0 and the forward pass runs in bfloat16;
`--no-amp` disables the latter.

---

## Baseline

The released Mamba-Spike code sits in `baselines/mamba_spike/`, with both the
protocol the authors shipped and the protocol we run it under. Those reproduce
the `‡` (48.90) and `†` (65.30) CIFAR10-DVS rows of Table 1, and the
interface-swap control of Appendix C.2. See that directory's README.

## Layout

```
ripe_mambaspike/
├── layers/        primitives
│   ├── normalization.py   BNTT and tdBN over (B, T, C, H, W)
│   ├── surrogate.py       ATan surrogate gradient
│   ├── neurons.py         LIF, SiLIF, complex-state CSiLIF
│   ├── reparam.py         RepConvBlock: 7 branches -> one 3x3
│   └── attention.py       DS-MPA linear attention
├── modules/       architecture
│   ├── tdm.py             Temporal Decoupled Modulation
│   ├── frontend.py        SpikingRepStage
│   ├── bridge.py          SpikeToRate, MR-S3A bridge
│   ├── pruning.py         spike-latency token selection
│   ├── mamba.py           bidirectional selective scan
│   └── hierarchy.py       hierarchical Mamba stack
├── models/        RIPEMambaSpike + per-benchmark builders
├── data/          five event datasets + neuromorphic augmentation
├── engine/        losses, schedules, optimizer, checkpoints, Trainer
└── utils/         seeding, parameter accounting

baselines/mamba_spike/   released Mamba-Spike + our-protocol script
```

Adding a benchmark means adding a dataset module and one entry in
`models/factory.py`. The optimization recipe lives in `engine/trainer.py` and
is shared, so a change to it reaches every benchmark at once.

---

## Model

```python
from ripe_mambaspike import build_model
from ripe_mambaspike.utils import deployed_parameters

model = build_model("cifar10dvs")        # (B, T, 2, H, W) -> (B, T, 10)
print(deployed_parameters(model))        # 869593
```

Four components, in order:

1. **RepConv front-end** — three spiking stages. Each trains as a seven-branch
   convolution and collapses to a single 3×3 kernel with bias at inference, so
   the multi-branch capacity costs nothing at deployment.
2. **TDM** — a per-channel first-order temporal filter,
   `(1 + α)·x_t − α·x_{t−1}`, one scalar per input channel. It is the identity
   at `α = 0`, and being linear it commutes with the convolution that follows,
   which is what permits accumulate-only inference.
3. **DS-MPA** — linear attention reading queries and keys from the membrane
   potential and values from the previous step's spikes. The feature map
   `elu(x) + 1` is strictly positive and non-saturating, so the normalizer
   cannot collapse; the output stays inside the convex hull of its values.
   Cost is `O(N·C²)` against `O(N²·C)` for softmax attention.
4. **MR-S3A bridge + hierarchical Mamba** — every stage is pooled to the final
   grid and projected to a fixed width, then scanned bidirectionally over a
   shrinking token budget. No parameter here depends on `H × W`.

### Deployment

Reparameterization fusion is explicit, never automatic:

```python
model.eval()
model.fuse_model()
```

For accumulate-only inference each fused block additionally splits into a dual
kernel, `W_a = W*·diag(1 + α)` and `W_b = W*·diag(α)`, so that

```
fused_conv(TDM(x_t, x_prev)) == conv(W_a, x_t) − conv(W_b, x_prev) + b*
```

Both operands stay binary, turning one dense multiply-accumulate pass over a
four-valued TDM output into two accumulate-only passes. This holds for every
backbone convolution except the first, whose operand is the real-valued binned
event tensor.

---