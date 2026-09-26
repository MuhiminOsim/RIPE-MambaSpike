"""Checkpoint saving and resumption.

Every run owns a directory under ``results/<run-name>/`` holding
``latest.pth``, ``best.pth`` and ``log.csv``. Resuming reads ``latest.pth``
and falls back to ``best.pth`` if it is missing or unreadable, so an
interrupted run restarts from the furthest recoverable point.
"""

import os

import torch

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Manage the checkpoint pair for one run.

    Args:
        run_dir: directory holding this run's artifacts.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.latest_path = os.path.join(run_dir, "latest.pth")
        self.best_path = os.path.join(run_dir, "best.pth")

    def save(self, model, optimizer, scheduler, epoch: int, best_acc: float,
             is_best: bool = False, **extra):
        """Write ``latest.pth``, and ``best.pth`` as well when ``is_best``."""
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        state.update(extra)
        torch.save(state, self.latest_path)
        if is_best:
            torch.save(state, self.best_path)

    def resume(self, model, optimizer=None, scheduler=None, map_location="cpu"):
        """Restore the furthest recoverable state.

        Returns ``(start_epoch, best_acc)``. A run with no checkpoint returns
        ``(0, 0.0)``, so ``--resume`` is safe to pass on a fresh run.
        """
        for path in (self.latest_path, self.best_path):
            if not os.path.exists(path):
                continue
            try:
                ckpt = torch.load(path, map_location=map_location, weights_only=False)
            except Exception as exc:
                print(f"[resume] {path} unreadable ({exc}); trying next")
                continue
            model.load_state_dict(ckpt["model"])
            if optimizer is not None and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", -1) + 1
            best_acc = ckpt.get("best_acc", 0.0)
            print(f"[resume] {path} at epoch {start_epoch} (best {best_acc:.2f}%)")
            return start_epoch, best_acc
        print("[resume] no checkpoint found; starting from scratch")
        return 0, 0.0
