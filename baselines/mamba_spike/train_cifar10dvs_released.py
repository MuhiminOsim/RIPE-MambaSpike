#!/usr/bin/env python3
"""
Train Mamba-Spike on CIFAR10-DVS using raw .pt files directly.
"""

import os
import sys
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.mamba_spike_real import create_mamba_spike_cifar10dvs
from models.mamba_spike_pooled import create_pooled_cifar10dvs


DATA_DIR = '/mnt/d/SNN Experiment/data/cifar-dvs'
# Linux filesystem cache — avoids slow WSL 9P filesystem overhead on D: drive
CACHE_DIR = os.path.expanduser('~/.cache/cifar10dvs')

import numpy as np


class CIFAR10DVSDataset(Dataset):
    """
    CIFAR10-DVS with RAM cache. Data is binary spikes stored as uint8 (~0.74 GB total),
    loaded entirely into RAM after a one-time build from the raw .pt files on D:.
    """

    def __init__(self, split='train'):
        os.makedirs(CACHE_DIR, exist_ok=True)
        data_path = os.path.join(CACHE_DIR, f'{split}_data.npy')
        labels_path = os.path.join(CACHE_DIR, f'{split}_labels.npy')

        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            self._build_cache(split, data_path, labels_path)

        print(f"[{split}] Loading cache into RAM...")
        # Keep as uint8 (~0.7 GB); cast to float32 per sample in __getitem__
        self.data = torch.from_numpy(np.load(data_path))   # uint8, (N, 10, 2, 128, 128)
        self.labels = np.load(labels_path)
        print(f"[{split}] In RAM: {self.data.shape}, {self.data.nbytes / 1e9:.2f} GB")

    def _build_cache(self, split, data_path, labels_path):
        folder = os.path.join(DATA_DIR, split)
        files = sorted(
            [f for f in os.listdir(folder) if f.endswith('.pt')],
            key=lambda x: int(x.replace('.pt', ''))
        )
        n = len(files)
        print(f"[{split}] Building cache for {n} samples (one-time, stored in {CACHE_DIR})...")
        # uint8: data is binary 0/1 spikes → 4× smaller than float32
        data_arr = np.zeros((n, 10, 2, 128, 128), dtype=np.uint8)
        labels_arr = np.zeros(n, dtype=np.int64)
        for i, fname in enumerate(tqdm(files, desc=f'Caching {split}')):
            sample = torch.load(os.path.join(folder, fname), weights_only=False)
            data_arr[i] = sample[0].permute(3, 0, 1, 2).numpy().astype(np.uint8)
            labels_arr[i] = int(sample[1].item())
        np.save(data_path, data_arr)
        np.save(labels_path, labels_arr)
        print(f"[{split}] Cache saved ({data_arr.nbytes / 1e9:.2f} GB on disk)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].float(), int(self.labels[idx])


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{max_epochs}")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.2f}%')

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Eval"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return total_loss / len(loader), 100. * correct / total


def find_latest_run(results_root='results'):
    """Return the most recent cifar10dvs results directory, or None."""
    if not os.path.isdir(results_root):
        return None
    runs = sorted([
        d for d in os.listdir(results_root)
        if d.startswith('cifar10dvs_') and os.path.isdir(os.path.join(results_root, d))
    ])
    return os.path.join(results_root, runs[-1]) if runs else None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--pooled', action='store_true',
                        help='Interface-swap control: replace the flatten-then-project '
                             'input_proj with adaptive-pool-then-project. Everything '
                             'else is the released model.')
    parser.add_argument('--pool-grid', type=int, default=8,
                        help='Pooled grid size (only with --pooled). 8 matches the '
                             'grid the released model already uses at 34x34.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in most recent results dir')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a specific checkpoint .pth file to resume from')
    args = parser.parse_args()

    device = setup_device()

    # Resolve checkpoint to resume from
    ckpt_path = None
    if args.resume_from:
        ckpt_path = args.resume_from
    elif args.resume:
        run_dir = find_latest_run()
        if run_dir:
            ckpt_path = os.path.join(run_dir, 'latest.pth')
            print(f"Resuming from: {ckpt_path}")
        else:
            print("No previous run found, starting fresh.")

    # Reuse existing run dir when resuming, otherwise create a new one
    if ckpt_path and os.path.exists(ckpt_path):
        out_dir = os.path.dirname(os.path.abspath(ckpt_path))
        # Load config to restore original hyperparams
        config_path = os.path.join(out_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                saved = json.load(f)
            args.batch_size = saved.get('batch_size', args.batch_size)
            args.lr = saved.get('lr', args.lr)
            args.epochs = saved.get('epochs', args.epochs)
    else:
        if ckpt_path:
            print(f"Checkpoint not found: {ckpt_path} — starting fresh.")
            ckpt_path = None
        tag = f"pooled{args.pool_grid}" if args.pooled else "released"
        out_dir = os.path.join('results', f"cifar10dvs_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

    print("\nLoading datasets...")
    train_ds = CIFAR10DVSDataset('train')
    test_ds = CIFAR10DVSDataset('test')
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    if args.pooled:
        model = create_pooled_cifar10dvs(pool_grid=args.pool_grid).to(device)
        arm = f'pooled(p={args.pool_grid})'
    else:
        model = create_mamba_spike_cifar10dvs().to(device)
        arm = 'released'
    total_params = sum(p.numel() for p in model.parameters())
    iface = sum(p.numel() for n_, p in model.named_parameters()
                if n_.startswith('input_proj'))
    print(f"Arm: {arm} | Model params: {total_params:,} | input_proj: {iface:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 1
    best_acc, best_epoch = 0.0, 0

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', ckpt.get('acc', 0.0))
        best_epoch = ckpt.get('best_epoch', ckpt['epoch'])
        # Fast-forward scheduler to match resumed epoch
        if 'scheduler' not in ckpt:
            for _ in range(ckpt['epoch']):
                scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']} (best: {best_acc:.2f}%)")

    config = {'batch_size': args.batch_size, 'lr': args.lr, 'epochs': args.epochs, 'device': str(device)}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    print(f"\n{'='*60}")
    print(f"Training: epochs {start_epoch}–{args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/acc', test_acc, epoch)

            is_best = test_acc > best_acc
            if is_best:
                best_acc, best_epoch = test_acc, epoch
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'opt': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_acc': best_acc, 'best_epoch': best_epoch},
                           os.path.join(out_dir, 'best.pth'))

            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'acc': test_acc, 'best_acc': best_acc, 'best_epoch': best_epoch},
                       os.path.join(out_dir, 'latest.pth'))

            print(f"\nEpoch {epoch}/{args.epochs} | LR={lr:.6f}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
            print(f"  Test:  loss={test_loss:.4f}, acc={test_acc:.2f}%")
            print(f"  Best:  {best_acc:.2f}% (epoch {best_epoch})")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        elapsed = time.time() - start
        print(f"\nDone. Best: {best_acc:.2f}% @ epoch {best_epoch} in {elapsed/3600:.2f}h")
        print(f"Results: {out_dir}")
        writer.close()


if __name__ == '__main__':
    main()
