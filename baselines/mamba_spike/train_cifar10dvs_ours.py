#!/usr/bin/env python3
"""
CIFAR10-DVS Dataset Training Script
Target Accuracy: 78.9% (from paper)
"""

import os
import sys
import time
import json
import random
import math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.mamba_spike import create_mamba_spike_cifar10dvs

# nda.py ships alongside this script in the released baseline directory
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from nda import NDA, polarity_flip, time_reversal
except ImportError:
    # Safe fallback if nda is not found
    print("Warning: nda module not found, proceeding without neuromorphic data augmentation.")
    class NDA:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    def polarity_flip(x): return x
    def time_reversal(x): return x

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy


class CosineWarmup:
    def __init__(self, optimizer, warmup, total, lr_init, lr_min):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.step_idx = 0

    def step(self):
        self.step_idx += 1
        if self.step_idx <= self.warmup:
            lr = self.lr_min + (self.lr_init - self.lr_min) * (self.step_idx / self.warmup)
        else:
            progress = (self.step_idx - self.warmup) / (self.total - self.warmup)
            lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def state_dict(self):
        return {'step_idx': self.step_idx}

    def load_state_dict(self, state_dict):
        self.step_idx = state_dict.get('step_idx', 0)


import copy

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        model_params  = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        for name, param in self.shadow.named_parameters():
            param.copy_(decay * param + (1.0 - decay) * model_params[name].detach())
        for name, buf in self.shadow.named_buffers():
            if buf.shape != model_buffers[name].shape:
                buf.resize_(model_buffers[name].shape)
            buf.copy_(model_buffers[name])

    def apply_to(self, model: nn.Module):
        orig_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow.state_dict())
        return orig_state

    def restore(self, model: nn.Module, orig_state):
        model.load_state_dict(orig_state)

    def state_dict(self):
        return {'shadow': self.shadow.state_dict(), 'num_updates': self.num_updates}

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict['shadow'])
        self.num_updates = state_dict['num_updates']





class CIFAR10DVSDataset(Dataset):
    """
    CIFAR10-DVS Dataset loader.
    Checks for pre-built cache in CACHE_DIR. If not found, builds cache from raw .pt files in data_dir.
    """
    def __init__(self, data_dir, split='train', cache_dir=None):
        self.is_train = (split == 'train')
        self.nda = NDA(n_ops=2, magnitude=0.5) if self.is_train else None

        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/cifar10dvs')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.data_path = os.path.join(cache_dir, f'{split}_data.npy')
        self.labels_path = os.path.join(cache_dir, f'{split}_labels.npy')
        
        if not os.path.exists(self.data_path) or not os.path.exists(self.labels_path):
            self._build_cache(data_dir, split, self.data_path, self.labels_path)
            
        print(f"[{split}] Loading cache into RAM...")
        self.data = torch.from_numpy(np.load(self.data_path))   # uint8, (N, 10, 2, 128, 128)
        self.labels = np.load(self.labels_path)
        print(f"[{split}] In RAM: {self.data.shape}, {self.data.nbytes / 1e9:.2f} GB")

    def _build_cache(self, data_dir, split, data_path, labels_path):
        # Resolve data_dir
        if not os.path.exists(data_dir):
            parent_fallback = os.path.join('..', data_dir)
            if os.path.exists(parent_fallback):
                data_dir = parent_fallback
            else:
                raise FileNotFoundError(f"CIFAR10-DVS dataset directory not found at '{data_dir}' or '{parent_fallback}'")
                
        folder = os.path.join(data_dir, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Split directory not found: {folder}")
            
        files = sorted(
            [f for f in os.listdir(folder) if f.endswith('.pt')],
            key=lambda x: int(x.replace('.pt', ''))
        )
        n = len(files)
        print(f"[{split}] Building cache for {n} samples (stored in {os.path.dirname(data_path)})...")
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
        data = self.data[idx].float() # (10, 2, 128, 128)
        
        if self.is_train:
            if random.random() > 0.5:
                data = torch.flip(data, dims=[3])  # H-flip (width is index 3)
            if random.random() > 0.5:
                data = polarity_flip(data)
            if random.random() > 0.5:
                data = time_reversal(data)
            if self.nda is not None:
                data = self.nda(data)
                
        return data, int(self.labels[idx])


class CIFAR10DVSTrainer:
    def __init__(self, batch_size=8, lr=0.001, max_epochs=200, data_dir='data/cifar-dvs', cache_dir=None, grad_accum=16, resume_path=None):
        # Paper target accuracy
        self.target_accuracy = 78.9
        self.start_epoch = 1

        # Setup device
        self.device = self._setup_device()

        # Training config
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_accum = grad_accum

        # Resolve output directory
        if resume_path is not None:
            if os.path.isdir(resume_path):
                self.output_dir = resume_path
            else:
                self.output_dir = os.path.dirname(resume_path)
        else:
            self.output_dir = os.path.join(
                "results",
                f"cifar10dvs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

        # Resolve data_dir fallback
        if not os.path.exists(data_dir):
            parent_fallback = os.path.join('..', data_dir)
            if os.path.exists(parent_fallback):
                data_dir = parent_fallback

        # Save config
        self.config = {
            "dataset": "cifar10dvs",
            "target_accuracy": self.target_accuracy,
            "batch_size": batch_size,
            "lr": lr,
            "max_epochs": max_epochs,
            "grad_accum": grad_accum,
            "device": str(self.device),
            "data_dir": data_dir
        }

        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Load dataset
        print("\n" + "="*70)
        print("CIFAR10-DVS Training")
        print("="*70)
        print(f"Target Accuracy: {self.target_accuracy}%")
        print(f"Max Epochs: {max_epochs}")
        print(f"Batch Size: {batch_size} (Grad Accum: {grad_accum} -> Eff BS: {batch_size * grad_accum})")
        print(f"Device: {self.device}")
        print("="*70 + "\n")

        print("Loading CIFAR10-DVS dataset...")
        self.num_classes = 10
        train_dataset = CIFAR10DVSDataset(data_dir=data_dir, split='train', cache_dir=cache_dir)
        test_dataset = CIFAR10DVSDataset(data_dir=data_dir, split='test', cache_dir=cache_dir)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # CRITICAL: Avoid DataLoader deadlock on Windows/WSL
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        print(f"✓ Dataset loaded: {len(self.train_loader)} train batches, "
              f"{len(self.test_loader)} test batches\n")

        # Create model
        print("Creating model...")
        self.model = create_mamba_spike_cifar10dvs().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model created: {total_params:,} parameters\n")

        # Loss and optimizer
        self.criterion_train = SoftTargetCrossEntropy()
        self.criterion_eval = nn.CrossEntropyLoss()
        self.mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=0.0, prob=0.5, label_smoothing=0.1, num_classes=10)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.03
        )

        # Learning rate scheduler
        self.scheduler = CosineWarmup(
            self.optimizer,
            warmup=10,
            total=max_epochs,
            lr_init=lr,
            lr_min=lr * 0.01
        )

        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_acc_ema = 0.0
        self.best_epoch_ema = 0
        
        # Exponential Moving Average
        self.ema = ModelEMA(self.model, decay=0.9999)
        self.start_time = time.time()

        # Resume logic
        if resume_path is not None:
            self._resume_checkpoint(resume_path)

    def _resume_checkpoint(self, resume_path):
        ckpt_files = []
        if os.path.isdir(resume_path):
            # Try latest, then best, then best_ema
            for name in ['checkpoint_latest.pth', 'checkpoint_best.pth', 'checkpoint_best_ema.pth']:
                ckpt_files.append(os.path.join(resume_path, name))
        else:
            ckpt_files.append(resume_path)

        loaded = False
        for ckpt_file in ckpt_files:
            if not os.path.exists(ckpt_file):
                continue
            print(f"Attempting to load checkpoint from '{ckpt_file}'...")
            try:
                # Load checkpoint
                checkpoint = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                
                # Check for needed keys
                if 'model_state_dict' not in checkpoint:
                    print(f"Warning: Checkpoint '{ckpt_file}' is missing model_state_dict. Skipping.")
                    continue
                
                # Run a dummy forward pass to allocate dynamic SSM buffers in the model
                print("Running a dummy forward pass to allocate dynamic SSM buffers...")
                dummy_x = torch.zeros((1, 10, 2, 128, 128), device=self.device)
                _ = self.model(dummy_x)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                if 'ema_state_dict' in checkpoint:
                    self.ema.load_state_dict(checkpoint['ema_state_dict'])
                else:
                    self.ema = ModelEMA(self.model, decay=0.9999)
                
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_acc = checkpoint.get('best_acc', 0.0)
                self.best_acc_ema = checkpoint.get('best_acc_ema', 0.0)
                self.best_epoch = checkpoint.get('epoch', 0) if self.best_acc > 0 else 0
                self.best_epoch_ema = checkpoint.get('epoch', 0) if self.best_acc_ema > 0 else 0
                
                print(f"✓ Successfully resumed from checkpoint '{ckpt_file}' at epoch {self.start_epoch}")
                print(f"  Best Acc: {self.best_acc:.2f}% | Best EMA Acc: {self.best_acc_ema:.2f}%")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Failed to load checkpoint '{ckpt_file}': {e}")
                print("Trying next available checkpoint...")

        if not loaded:
            print("⚠ Error: Could not resume from any valid checkpoint in the resume path.")
            print("Starting training from scratch instead.")

    def _setup_device(self):
        """Setup computing device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                _ = torch.zeros(1).to(device)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"\n✓ GPU detected: {gpu_name}")
                return device
            except:
                print("\n⚠ CUDA available but initialization failed, using CPU")
                return torch.device("cpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            try:
                _ = torch.zeros(1).to(device)
                print(f"\n✓ Apple Silicon GPU (MPS) detected")
                return device
            except:
                print("\n⚠ MPS available but initialization failed, using CPU")
                return torch.device("cpu")
        else:
            print("\n⚠ No GPU detected, using CPU")
            return torch.device("cpu")

    def _event_cutmix(self, x, y, alpha=0.4):
        if random.random() > 0.5 or x.size(0) < 2:
            return x, y
        lam = float(np.random.beta(alpha, alpha))
        B, T, C, H, W = x.shape
        cut_h = int(H * math.sqrt(1 - lam))
        cut_w = int(W * math.sqrt(1 - lam))
        cy = random.randint(0, H - cut_h) if cut_h > 0 else 0
        cx = random.randint(0, W - cut_w) if cut_w > 0 else 0
        perm = torch.randperm(B, device=x.device)
        x = x.clone()
        x[:, :, :, cy:cy+cut_h, cx:cx+cut_w] = x[perm][:, :, :, cy:cy+cut_h, cx:cx+cut_w]
        lam_actual = 1.0 - (cut_h * cut_w) / (H * W)
        if y.ndim == 1:
            y = torch.nn.functional.one_hot(y, num_classes=10).float()
        y = lam_actual * y + (1 - lam_actual) * y[perm]
        return x, y

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            if data.size(0) % 2 == 0:
                # Mutually exclusive Mixup and Cutmix (either mixup OR cutmix, not both sequentially)
                if random.random() > 0.5:
                    data, target = self.mixup_fn(data, target)
                else:
                    data, target = self._event_cutmix(data, target)

            # Ensure target is soft targets for SoftTargetCrossEntropy
            if target.ndim == 1:
                target = torch.nn.functional.one_hot(target, num_classes=10).float()
                target = target * (1.0 - 0.1) + 0.1 / 10.0

            output = self.model(data)
            loss_full = self.criterion_train(output, target)
            loss = loss_full / self.grad_accum
            loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update(self.model)

            total_loss += loss_full.item()
            with torch.no_grad():
                pred = output.argmax(dim=1)
                tgt = target.argmax(dim=1) if target.ndim > 1 else target
                correct += pred.eq(tgt).sum().item()
                total += data.size(0)

            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{loss_full.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total

        return train_loss, train_acc

    def evaluate(self, model=None):
        """Evaluate on test set."""
        if model is None:
            model = self.model
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion_eval(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = total_loss / len(self.test_loader)
        test_acc = 100. * correct / total

        return test_loss, test_acc

    def save_checkpoint(self, epoch, test_acc, is_best=False, test_acc_ema=0.0, is_best_ema=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'test_acc': test_acc,
            'test_acc_ema': test_acc_ema,
            'best_acc': self.best_acc,
            'best_acc_ema': self.best_acc_ema,
            'config': self.config
        }

        # Always save latest
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best raw model
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {test_acc:.2f}%")

        # Save best EMA model
        if is_best_ema:
            best_ema_path = os.path.join(self.output_dir, 'checkpoint_best_ema.pth')
            checkpoint_ema = checkpoint.copy()
            checkpoint_ema['model_state_dict'] = self.ema.shadow.state_dict()
            torch.save(checkpoint_ema, best_ema_path)
            print(f"  ✓ Saved best EMA model: {test_acc_ema:.2f}%")

    def train(self):
        """Main training loop with early stopping."""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70 + "\n")

        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                # Update scheduler (warmup / cosine annealing)
                current_lr = self.scheduler.step()

                # Train
                train_loss, train_acc = self.train_epoch(epoch)

                # Evaluate raw model
                test_loss, test_acc = self.evaluate(self.model)

                # Evaluate EMA model
                test_loss_ema, test_acc_ema = self.evaluate(self.ema.shadow)

                # Log to TensorBoard
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/acc', train_acc, epoch)
                self.writer.add_scalar('test/loss', test_loss, epoch)
                self.writer.add_scalar('test/acc', test_acc, epoch)
                self.writer.add_scalar('test/loss_ema', test_loss_ema, epoch)
                self.writer.add_scalar('test/acc_ema', test_acc_ema, epoch)
                self.writer.add_scalar('train/lr', current_lr, epoch)

                # Print results
                print(f"\nEpoch {epoch}/{self.max_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% (EMA: {test_acc_ema:.2f}%)")
                print(f"  Learning Rate: {current_lr:.6f}")

                # Check if best
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self.best_epoch = epoch

                is_best_ema = test_acc_ema > self.best_acc_ema
                if is_best_ema:
                    self.best_acc_ema = test_acc_ema
                    self.best_epoch_ema = epoch

                # Save checkpoint
                self.save_checkpoint(epoch, test_acc, is_best, test_acc_ema, is_best_ema)

                # Progress update
                print(f"  Best so far: {self.best_acc:.2f}% (Epoch {self.best_epoch}) | EMA Best: {self.best_acc_ema:.2f}% (Epoch {self.best_epoch_ema})")
                print(f"  Target: {self.target_accuracy:.2f}%")
                print()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")

        finally:
            # Final summary
            elapsed = time.time() - self.start_time
            print("\n" + "="*70)
            print("Training Complete")
            print("="*70)
            print(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
            print(f"Best EMA Test Accuracy: {self.best_acc_ema:.2f}% (Epoch {self.best_epoch_ema})")
            print(f"Target Accuracy: {self.target_accuracy}%")
            print(f"Gap to target: {self.target_accuracy - max(self.best_acc, self.best_acc_ema):.2f}%")
            print(f"Total training time: {elapsed/3600:.2f} hours")
            print(f"Results saved to: {self.output_dir}")
            print("="*70 + "\n")

            self.writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train on CIFAR10-DVS dataset')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8, conservative for stability)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum epochs (default: 200)')
    parser.add_argument('--grad-accum', type=int, default=16,
                       help='Gradient accumulation steps (default: 16)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file or results directory to resume from')
    parser.add_argument('--data-dir', type=str, default='data/cifar-dvs',
                       help='Path to CIFAR10-DVS dataset (default: data/cifar-dvs)')
    args = parser.parse_args()

    trainer = CIFAR10DVSTrainer(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        data_dir=args.data_dir,
        grad_accum=args.grad_accum,
        resume_path=args.resume
    )

    trainer.train()


if __name__ == "__main__":
    main()
