# Mamba-Spike baseline

The released [Mamba-Spike](https://arxiv.org/abs/2408.11823) code, plus the
script that runs it under our training recipe. Table 1 of the paper reports
this baseline three ways, and the two scripts here reproduce the two that are
ours to reproduce.

| Table 1 marker | Meaning | Script | CIFAR10-DVS |
|---|---|---|---|
| `*` | the authors' own published figure | — (quoted, not rerun) | 92.50 |
| `‡` | released code, **its own** protocol | `train_cifar10dvs_released.py` | **48.90** |
| `†` | released code, **our** protocol | `train_cifar10dvs_ours.py` | **65.30** |

Our recipe improves the released code by 16.40 pp on CIFAR10-DVS and on every
other shared benchmark, which is why the paper rests its comparison on `†`
rather than on the published figure: the gap to 92.50 is not under-training on
our part.

## What differs between the two scripts

Only the training protocol. Both build the same architecture at the same size,
which you can check without downloading any data:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from models.mamba_spike_real import create_mamba_spike_cifar10dvs as released
from models.mamba_spike import create_mamba_spike_cifar10dvs as ours
for tag, f in (('released', released), ('ours', ours)):
    m = f(); tot = sum(p.numel() for p in m.parameters())
    ip = sum(p.numel() for n,p in m.named_parameters() if 'input_proj' in n)
    print(f'{tag:9} {tot:,} total, {ip:,} in input_proj ({100*ip/tot:.1f}%)')"
```

```
released  36,252,874 total, 33,554,688 in input_proj (92.6%)
ours      36,249,802 total, 33,554,688 in input_proj (92.6%)
```

Both round to the 36.25 M the paper quotes, and both put 92.6% of parameters
in `input_proj`, the single resolution-dependent projection the paper is
about. The 3,072-parameter difference between the two modules is in the head
and does not touch that projection.

| | `train_cifar10dvs_released.py` | `train_cifar10dvs_ours.py` |
|---|---|---|
| Optimizer | AdamW, wd 1e-4 | AdamW, wd 0.03 |
| Schedule | cosine | warm-up + cosine |
| Batch | 32 | 8 x 16 accumulation (eff. 128) |
| Augmentation | none | NDA, polarity flip, h-flip |
| Mixup / CutMix | none | Mixup 0.4, p 0.5, smoothing 0.1 |
| EMA | none | 0.9999 |

## Running

```bash
python train_cifar10dvs_released.py --epochs 200 --data-root /path/to/cifar-dvs
python train_cifar10dvs_ours.py     --epochs 300 --data-root /path/to/cifar-dvs
```

`train_dvsgesture.py` and `train_nmnist.py` are the released scripts for the
other two shared benchmarks, included unmodified.

## Interface-swap control

`models/mamba_spike_pooled.py` is **ours, not released**. It is the control in
Appendix C.2: the released model with its flatten-then-project bridge replaced
by pool-then-project, everything else held fixed. Run it through the released
protocol with `--pooled`:

```bash
python train_cifar10dvs_released.py --pooled --pool-grid 8 --epochs 200 \
       --data-root /path/to/cifar-dvs
```

## Provenance

`models/`, `train_cifar10dvs_released.py` (upstream `train_cifar10dvs_direct.py`),
`train_dvsgesture.py` and `train_nmnist.py` are the released code. Two changes
were made, both mechanical:

- `train_cifar10dvs_ours.py` and `train_dvsgesture.py` located `nda.py` one
  directory above their original home; they now look in their own directory,
  where `nda.py` ships.
- `train_cifar10dvs_released.py` gained `--pooled` / `--pool-grid` for the
  interface-swap control. Without those flags its behaviour is unchanged.

Upstream licence in `LICENSE_mamba_spike`.
