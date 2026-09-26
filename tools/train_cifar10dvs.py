import os

from timm.data.mixup import Mixup

from _common import base_parser, setup
from ripe_mambaspike.data.augment import make_cutmix
from ripe_mambaspike.data.cifar10dvs import CIFAR10DVS, NUM_CLASSES


def main():
    parser = base_parser("cifar10dvs", epochs=300, batch_size=32, lr=1e-3, T=10,
                         grad_accum=1, weight_decay=0.03, warmup=15,
                         tet_lambda=1e-2, sgc_lambda=0.0, l1_lambda=0.0,
                         alpha_end=3.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--time-reversal", action="store_true", default=False)
    parser.add_argument("--nda-ops", type=int, default=2)
    parser.add_argument("--nda-magnitude", type=float, default=0.5)
    args = parser.parse_args()

    train_ds = CIFAR10DVS(
        os.path.join(args.data_root, "train"), train=True, resolution=args.resolution,
        T=args.T, cache_dir=args.cache_dir, time_reversal=args.time_reversal,
        nda_ops=args.nda_ops, nda_magnitude=args.nda_magnitude,
    )
    test_ds = CIFAR10DVS(
        os.path.join(args.data_root, "test"), train=False, resolution=args.resolution,
        T=args.T, cache_dir=args.cache_dir,
    )

    clean_ds = CIFAR10DVS(
        os.path.join(args.data_root, "train"), train=False, resolution=args.resolution,
        T=args.T, cache_dir=args.cache_dir, cache_split="train",
    )

    mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=0.0, prob=0.5,
                     label_smoothing=0.05, num_classes=NUM_CLASSES)

    trainer = setup("cifar10dvs", args, train_ds, test_ds, mixup_fn=mixup_fn,
                    cutmix_fn=make_cutmix(NUM_CLASSES, alpha=0.4), clean_ds=clean_ds)
    print(f"best test accuracy {trainer.fit():.2f}%")


if __name__ == "__main__":
    main()
