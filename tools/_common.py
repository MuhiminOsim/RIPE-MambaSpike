"""Shared argument parsing and run setup for the training scripts."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripe_mambaspike.engine import Trainer
from ripe_mambaspike.models import build_model
from ripe_mambaspike.utils import count_parameters, deployed_parameters, seed_everything

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def base_parser(benchmark, *, epochs, batch_size, lr, T, grad_accum=1,
                weight_decay=0.05, lr_min=1e-6, warmup=10, workers=4,
                tet_lambda=5e-3, sgc_lambda=1.0, l1_lambda=1e-4,
                alpha_end=4.0):
    parser = argparse.ArgumentParser(description=f"Train RIPE-MambaSpike on {benchmark}")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=benchmark)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--lr-min", type=float, default=lr_min)
    parser.add_argument("--weight-decay", type=float, default=weight_decay)
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--grad-accum", type=int, default=grad_accum)
    parser.add_argument("--T", type=int, default=T)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--l1-lambda", type=float, default=l1_lambda)
    parser.add_argument("--sgc-lambda", type=float, default=sgc_lambda)
    parser.add_argument("--tet-lambda", type=float, default=tet_lambda)
    parser.add_argument("--alpha-start", type=float, default=2.0)
    parser.add_argument("--alpha-end", type=float, default=alpha_end)
    parser.add_argument("--norm-type", type=str, default="bntt", choices=["bntt", "tdbn"])
    parser.add_argument("--no-sgc", dest="sgc", action="store_false", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--no-bn-recalib", dest="bn_recalib",
                        action="store_false", default=True)
    parser.add_argument("--bn-recalib-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser


def make_loaders(train_ds, test_ds, args):
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
    )
    return train_loader, test_loader


def setup(benchmark, args, train_ds, test_ds, *, mixup_fn=None, cutmix_fn=None,
          model_overrides=None, clean_ds=None):
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    overrides = dict(T=args.T, top_k=args.top_k, norm_type=args.norm_type, sgc=args.sgc)
    if model_overrides:
        overrides.update(model_overrides)
    model = build_model(benchmark, **overrides)

    print(f"benchmark       {benchmark}")
    print(f"train / test    {len(train_ds)} / {len(test_ds)}")
    print(f"params train    {count_parameters(model):,}")
    print(f"params deployed {deployed_parameters(model):,}")

    train_loader, test_loader = make_loaders(train_ds, test_ds, args)
    clean_loader = None
    if args.bn_recalib and clean_ds is not None:
        clean_loader = DataLoader(
            clean_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=max(1, args.workers // 2), pin_memory=True,
        )
    run_dir = os.path.join(RESULTS_ROOT, args.run_name)

    trainer = Trainer(
        model, train_loader, test_loader, run_dir,
        epochs=args.epochs, lr=args.lr, lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        warmup=args.warmup, grad_accum=args.grad_accum,
        l1_lambda=args.l1_lambda, sgc_lambda=args.sgc_lambda,
        tet_lambda=args.tet_lambda, amp=args.amp, device=device,
        alpha_start=args.alpha_start, alpha_end=args.alpha_end,
        mixup_fn=mixup_fn, cutmix_fn=cutmix_fn,
        clean_loader=clean_loader, bn_recalib_batches=args.bn_recalib_batches,
    )
    if args.resume:
        trainer.resume()
    print(f"results         {run_dir}")
    return trainer
