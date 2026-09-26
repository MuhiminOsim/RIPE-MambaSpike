#!/usr/bin/env python3
"""
DVS Gesture Dataset Training Script
Target Accuracy: 96.8% (from paper)
"""

import os
import sys
import time
import json
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tonic.transforms as tonic_transforms
from tqdm import tqdm

from models.mamba_spike import create_mamba_spike_dvsgesture

# Add parent directory to sys.path to access nda and other root level utilities if needed
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from nda import NDA, polarity_flip
except ImportError:
    # Safe fallback if nda is not found
    print("Warning: nda module not found, proceeding without neuromorphic data augmentation.")
    class NDA:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x
    def polarity_flip(x): return x

SENSOR_SIZE = (128, 128, 2)
_EV_DTYPE   = np.dtype([('x', np.int16), ('y', np.int16),
                         ('p', bool),     ('t', np.int64)])

def _to_structured(raw):
    """Convert (N,4) float64 [x,y,p,t_ms] → structured event array."""
    evs = np.empty(len(raw), dtype=_EV_DTYPE)
    evs['x'] = raw[:, 0].astype(np.int16)
    evs['y'] = raw[:, 1].astype(np.int16)
    evs['p'] = raw[:, 2].astype(bool)
    evs['t'] = (raw[:, 3] * 1000).astype(np.int64)   # ms → µs
    return evs


def _random_time_crop(raw, window_ms=1500.0):
    if len(raw) < 10: return raw
    t = raw[:, 3]
    t_min, t_max = float(t.min()), float(t.max())
    total = t_max - t_min
    if total <= window_ms:
        return raw
    start = t_min + random.uniform(0.0, total - window_ms)
    mask = (t >= start) & (t <= start + window_ms)
    cropped = raw[mask].copy()
    if len(cropped) > 0:
        cropped[:, 3] -= start
    return cropped if len(cropped) >= 10 else raw


def _center_time_crop(raw, window_ms=1500.0):
    if len(raw) < 10: return raw
    t = raw[:, 3]
    t_min, t_max = float(t.min()), float(t.max())
    total = t_max - t_min
    if total <= window_ms:
        return raw
    start = t_min + (total - window_ms) / 2.0
    mask = (t >= start) & (t <= start + window_ms)
    cropped = raw[mask].copy()
    if len(cropped) > 0:
        cropped[:, 3] -= start
    return cropped if len(cropped) >= 10 else raw


def _spatial_roll(data, max_px=12):
    """Random spatial translation by rolling H and W."""
    sh = random.randint(-max_px, max_px)
    sw = random.randint(-max_px, max_px)
    return torch.roll(data, shifts=(sh, sw), dims=(2, 3))   # (T,2,H,W)


class DVSGestureDataset(Dataset):
    def __init__(self, data_dir, train, T=16, window_ms=1500.0):
        self.is_train  = train
        self.window_ms = window_ms
        self.nda       = NDA(n_ops=1, magnitude=0.3) if train else None

        subset   = 'ibmGestureTrain' if train else 'ibmGestureTest'
        root     = os.path.join(data_dir, subset)
        
        self.frame_tf = tonic_transforms.ToFrame(
            sensor_size=SENSOR_SIZE, n_time_bins=T)

        self.samples = []
        for rec_dir in sorted(os.listdir(root)):
            rec_path = os.path.join(root, rec_dir)
            if not os.path.isdir(rec_path):
                continue
            for fname in os.listdir(rec_path):
                if fname.endswith('.npy'):
                    self.samples.append(
                        (os.path.join(rec_path, fname), int(fname[:-4])))

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, raw):
        evs    = _to_structured(raw)
        frames = self.frame_tf(evs)
        return torch.from_numpy(frames.astype('float32')).clamp(0.0, 1.0)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        raw = np.load(path)

        if self.is_train:
            raw  = _random_time_crop(raw, window_ms=self.window_ms)
            data = self._load_frames(raw)
            
            if random.random() > 0.5:
                data = polarity_flip(data)
            if random.random() > 0.5:
                data = _spatial_roll(data)
                
            data = self.nda(data)
            return data, label

        raw = _center_time_crop(raw, window_ms=self.window_ms)
        return self._load_frames(raw), label


class DVSGestureTrainer:
    def __init__(self, batch_size=16, lr=0.001, max_epochs=200, data_dir='data/DvsGesturePreprocessed', T=16, window_ms=1500.0):
        # Paper target accuracy
        self.target_accuracy = 96.8

        # Setup device
        self.device = self._setup_device()

        # Training config
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs

        # Create output directory
        self.output_dir = os.path.join(
            "results",
            f"dvsgesture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        self.config = {
            "dataset": "dvsgesture",
            "target_accuracy": self.target_accuracy,
            "batch_size": batch_size,
            "lr": lr,
            "max_epochs": max_epochs,
            "device": str(self.device),
            "data_dir": data_dir,
            "T": T,
            "window_ms": window_ms
        }

        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Load dataset
        print("\n" + "="*70)
        print("DVS Gesture Training")
        print("="*70)
        print(f"Target Accuracy: {self.target_accuracy}%")
        print(f"Max Epochs: {max_epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")

        # Resolve data_dir
        if not os.path.exists(data_dir):
            parent_fallback = os.path.join('..', data_dir)
            if os.path.exists(parent_fallback):
                data_dir = parent_fallback
            else:
                raise FileNotFoundError(f"DvsGesturePreprocessed dataset directory not found at '{data_dir}' or '{parent_fallback}'")

        print("Loading DVS Gesture dataset...")
        self.num_classes = 11
        train_dataset = DVSGestureDataset(data_dir=data_dir, train=True, T=T, window_ms=window_ms)
        test_dataset = DVSGestureDataset(data_dir=data_dir, train=False, T=T, window_ms=window_ms)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid DataLoader issues
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid DataLoader issues
            pin_memory=True
        )
        print(f"✓ Dataset loaded: {len(self.train_loader)} train batches, "
              f"{len(self.test_loader)} test batches\n")

        # Create model
        print("Creating model...")
        self.model = create_mamba_spike_dvsgesture().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model created: {total_params:,} parameters\n")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.0001
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01
        )

        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.start_time = time.time()

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

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total

        return train_loss, train_acc

    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = total_loss / len(self.test_loader)
        test_acc = 100. * correct / total

        return test_loss, test_acc

    def save_checkpoint(self, epoch, test_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'test_acc': test_acc,
            'best_acc': self.best_acc,
            'config': self.config
        }

        # Always save latest
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {test_acc:.2f}%")

    def train(self):
        """Main training loop with early stopping."""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70 + "\n")

        try:
            for epoch in range(1, self.max_epochs + 1):
                # Train
                train_loss, train_acc = self.train_epoch(epoch)

                # Evaluate
                test_loss, test_acc = self.evaluate()

                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log to TensorBoard
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/acc', train_acc, epoch)
                self.writer.add_scalar('test/loss', test_loss, epoch)
                self.writer.add_scalar('test/acc', test_acc, epoch)
                self.writer.add_scalar('train/lr', current_lr, epoch)

                # Print results
                print(f"\nEpoch {epoch}/{self.max_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")

                # Check if best
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self.best_epoch = epoch

                # Save checkpoint
                self.save_checkpoint(epoch, test_acc, is_best)

                # Progress update
                print(f"  Best so far: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
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
            print(f"Target Accuracy: {self.target_accuracy}%")
            print(f"Gap to target: {self.target_accuracy - self.best_acc:.2f}%")
            print(f"Total training time: {elapsed/3600:.2f} hours")
            print(f"Results saved to: {self.output_dir}")
            print("="*70 + "\n")

            self.writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train on DVS Gesture dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum epochs (default: 200)')
    parser.add_argument('--data-dir', type=str, default='data/DvsGesturePreprocessed',
                       help='Path to DvsGesturePreprocessed dataset (default: data/DvsGesturePreprocessed)')
    parser.add_argument('--T', type=int, default=16,
                       help='Number of time bins (default: 16)')
    parser.add_argument('--window-ms', type=float, default=1500.0,
                       help='Fixed duration window in ms for temporal cropping (default: 1500.0)')
    args = parser.parse_args()

    trainer = DVSGestureTrainer(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        data_dir=args.data_dir,
        T=args.T,
        window_ms=args.window_ms
    )

    trainer.train()


if __name__ == "__main__":
    main()
