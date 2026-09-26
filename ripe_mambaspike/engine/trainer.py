"""Training loop shared by every benchmark script.

One :class:`Trainer` drives all five benchmarks. What differs between them
lives in the dataset module and the model configuration, never here, so a
change to the optimization recipe lands on every benchmark at once.
"""

import math

import torch
import torch.nn as nn
from tqdm import tqdm

from .bn_recalib import recalibrate_bn, restore_bn, snapshot_bn
from .checkpoint import CheckpointManager
from .logger import CSVLogger
from .losses import sgc_loss, tet_loss
from .optim import build_optimizer
from .scheduler import CosineWarmup, continuous_weight, surrogate_alpha

__all__ = ["Trainer", "TRAIN_FIELDS"]


TRAIN_FIELDS = (
    "epoch", "lr", "alpha",
    "train_loss", "train_loss_tet", "train_loss_sgc", "train_loss_l1",
    "train_acc", "test_loss", "test_acc", "best_acc",
    "grad_norm", "spike_rate",
)


class Trainer:
    """Drive training, evaluation, logging and checkpointing for one run.

    Args:
        model: the network.
        train_loader, test_loader: data loaders.
        run_dir: directory for ``latest.pth``, ``best.pth`` and ``log.csv``.
        epochs: total epochs.
        lr: peak learning rate.
        lr_min: floor of the cosine decay.
        weight_decay: decay applied to the decaying parameter group.
        warmup: warm-up epochs.
        grad_accum: micro-batches per optimizer step.
        l1_lambda: weight on the firing-rate penalty.
        sgc_lambda: weight on surrogate-gradient consistency.
        tet_lambda: weight on the TET consistency term.
        clip_grad: gradient-norm clip.
        alpha_start, alpha_end: surrogate sharpness at the first and last
            epoch. The end point is per-benchmark; see
            :func:`~ripe_mambaspike.engine.scheduler.surrogate_alpha`.
        amp: run the forward pass in bfloat16.
        device: torch device.
        mixup_fn: optional callable applied to ``(x, y)`` before the model.
        cutmix_fn: optional callable applied after ``mixup_fn``.
        clean_loader: un-augmented view of the training split. When given,
            BatchNorm statistics are recalibrated on it before each
            evaluation and restored afterwards.
        bn_recalib_batches: batches drawn from ``clean_loader``.
    """

    def __init__(self, model, train_loader, test_loader, run_dir, *, epochs=300,
                 lr=1e-3, lr_min=1e-6, weight_decay=0.05, warmup=10, grad_accum=1,
                 l1_lambda=1e-4, sgc_lambda=1.0, tet_lambda=1e-2,
                 clip_grad=1.0, amp=True, device=None,
                 alpha_start=2.0, alpha_end=4.0,
                 mixup_fn=None, cutmix_fn=None,
                 clean_loader=None, bn_recalib_batches=50):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.grad_accum = grad_accum
        self.l1_lambda = l1_lambda
        self.sgc_lambda = sgc_lambda
        self.tet_lambda = tet_lambda
        self.clip_grad = clip_grad
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.amp = amp
        self.mixup_fn = mixup_fn
        self.cutmix_fn = cutmix_fn
        self.clean_loader = clean_loader
        self.bn_recalib_batches = bn_recalib_batches

        self.optimizer = build_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineWarmup(self.optimizer, warmup, epochs, lr, lr_min)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp and self.device.type == "cuda")

        self.ckpt = CheckpointManager(run_dir)
        self.logger = CSVLogger(run_dir, TRAIN_FIELDS)
        self.start_epoch = 0
        self.best_acc = 0.0

    def resume(self):
        """Restore model, optimizer, scheduler, epoch and best accuracy."""
        self.start_epoch, self.best_acc = self.ckpt.resume(
            self.model, self.optimizer, self.scheduler, map_location=self.device
        )
        return self.start_epoch

    def _autocast(self):
        return torch.amp.autocast(
            "cuda", enabled=self.amp and self.device.type == "cuda", dtype=torch.bfloat16
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_alpha(
            surrogate_alpha(epoch, self.epochs, self.alpha_start, self.alpha_end))
        cont_w = continuous_weight(epoch, self.epochs)

        totals = dict(loss=0.0, tet=0.0, sgc=0.0, l1=0.0)
        correct = total = 0
        grad_norms, spike_rates = [], []

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if self.mixup_fn is not None and x.size(0) % 2 == 0:
                x, y = self.mixup_fn(x, y)
            if self.cutmix_fn is not None:
                x, y = self.cutmix_fn(x, y)

            with self._autocast():
                logits_d, logits_c, spike_rate = self.model(x)
                loss_tet = tet_loss(logits_d, y, lamb=self.tet_lambda)
                loss_tet_c = tet_loss(logits_c, y, lamb=self.tet_lambda)
                loss_sgc = sgc_loss(logits_d, logits_c)
                loss_l1 = spike_rate * self.l1_lambda
                loss = ((1.0 - cont_w) * loss_tet + cont_w * loss_tet_c
                        + self.sgc_lambda * loss_sgc + loss_l1)

            self.scaler.scale(loss / self.grad_accum).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                if math.isfinite(float(norm)):
                    grad_norms.append(float(norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            totals["loss"] += loss.item()
            totals["tet"] += loss_tet.item()
            totals["sgc"] += loss_sgc.item()
            totals["l1"] += float(loss_l1)
            spike_rates.append(float(spike_rate))

            with torch.no_grad():
                pred = logits_d.mean(1).argmax(1)
                target = y.argmax(1) if y.ndim > 1 else y
                correct += pred.eq(target).sum().item()
                total += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.0 * correct / max(1, total):.2f}")

        n = max(1, len(self.train_loader))
        return {
            "train_loss": totals["loss"] / n,
            "train_loss_tet": totals["tet"] / n,
            "train_loss_sgc": totals["sgc"] / n,
            "train_loss_l1": totals["l1"] / n,
            "train_acc": 100.0 * correct / max(1, total),
            "grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
            "spike_rate": sum(spike_rates) / max(1, len(spike_rates)),
        }

    @torch.no_grad()
    def evaluate(self):
        """Single-pass evaluation, after recalibrating BatchNorm on clean data.

        The recalibration is undone before returning, so the statistics the
        next epoch trains with are the ones training produced.
        """
        snapshot = None
        if self.clean_loader is not None:
            snapshot = snapshot_bn(self.model)
            recalibrate_bn(self.model, self.clean_loader, self.device,
                           n_batches=self.bn_recalib_batches, amp=self.amp)
        try:
            return self._evaluate_pass()
        finally:
            if snapshot is not None:
                restore_bn(self.model, snapshot)

    @torch.no_grad()
    def _evaluate_pass(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_sum, correct, total, batches = 0.0, 0, 0, 0
        for x, y in tqdm(self.test_loader, desc="eval", leave=False):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            loss_sum += tet_loss(logits, y, lamb=self.tet_lambda, criterion=criterion).item()
            correct += logits.mean(1).argmax(1).eq(y).sum().item()
            total += x.size(0)
            batches += 1
        return {
            "test_loss": loss_sum / max(1, batches),
            "test_acc": 100.0 * correct / max(1, total),
        }

    def fit(self):
        """Run training to completion, logging and checkpointing each epoch."""
        for epoch in range(self.start_epoch, self.epochs):
            lr = self.scheduler.step()
            metrics = self.train_epoch(epoch)
            metrics.update(self.evaluate())

            is_best = metrics["test_acc"] > self.best_acc
            self.best_acc = max(self.best_acc, metrics["test_acc"])

            metrics.update(epoch=epoch, lr=lr,
                           alpha=surrogate_alpha(epoch, self.epochs,
                                                 self.alpha_start, self.alpha_end),
                           best_acc=self.best_acc)
            self.logger.append(metrics)
            self.ckpt.save(self.model, self.optimizer, self.scheduler,
                           epoch, self.best_acc, is_best=is_best)

            print(f"epoch {epoch:3d} | lr {lr:.2e} | "
                  f"train {metrics['train_acc']:6.2f}% | "
                  f"test {metrics['test_acc']:6.2f}% | best {self.best_acc:6.2f}%")
        return self.best_acc
