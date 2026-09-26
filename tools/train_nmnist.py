from timm.data.mixup import Mixup

from _common import base_parser, setup
from ripe_mambaspike.data.augment import make_cutmix
from ripe_mambaspike.data.nmnist import NMNIST, NUM_CLASSES


def main():
    parser = base_parser("nmnist", epochs=300, batch_size=64, lr=1e-3, T=10,
                         grad_accum=2, weight_decay=0.05, warmup=10,
                         tet_lambda=5e-3, sgc_lambda=1.0, l1_lambda=1e-4)
    parser.add_argument("--nda-ops", type=int, default=2)
    parser.add_argument("--nda-magnitude", type=float, default=0.3)
    args = parser.parse_args()

    train_ds = NMNIST(args.data_root, train=True, T=args.T,
                      nda_ops=args.nda_ops, nda_magnitude=args.nda_magnitude)
    test_ds = NMNIST(args.data_root, train=False, T=args.T)
    clean_ds = NMNIST(args.data_root, train=True, T=args.T, clean=True)

    mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=0.0, prob=0.5,
                     label_smoothing=0.05, num_classes=NUM_CLASSES)

    trainer = setup("nmnist", args, train_ds, test_ds, mixup_fn=mixup_fn,
                    cutmix_fn=make_cutmix(NUM_CLASSES, alpha=0.2), clean_ds=clean_ds)
    print(f"best test accuracy {trainer.fit():.2f}%")


if __name__ == "__main__":
    main()
