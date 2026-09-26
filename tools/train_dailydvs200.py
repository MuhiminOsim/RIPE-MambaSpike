from timm.data.mixup import Mixup

from _common import base_parser, setup
from ripe_mambaspike.data.dailydvs200 import NUM_CLASSES, DailyDVS200, build_split


def main():
    parser = base_parser("dailydvs200", epochs=250, batch_size=32, lr=5e-4, T=10,
                         grad_accum=1, weight_decay=0.02, warmup=10,
                         tet_lambda=5e-3, sgc_lambda=0.0, l1_lambda=1e-4)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--bin-mode", type=str, default="uniform",
                        choices=["uniform", "window", "count"])
    parser.add_argument("--label-json", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    args = parser.parse_args()

    splits, meta = build_split(args.data_root, label_json=args.label_json,
                               split_dir=args.split_dir)
    print(f"split source    {meta.get('source', 'derived')}")

    common = dict(T=args.T, H=args.resolution, W=args.resolution,
                  bin_mode=args.bin_mode, cache_dir=args.cache_dir)
    train_ds = DailyDVS200(splits["train"], augment=True, **common)
    test_ds = DailyDVS200(splits["test"], augment=False, **common)
    clean_ds = DailyDVS200(splits["train"], augment=False, **common)

    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.0, prob=0.5,
                     label_smoothing=0.1, num_classes=NUM_CLASSES)

    benchmark = "dailydvs200_lite" if args.lite else "dailydvs200"
    trainer = setup(benchmark, args, train_ds, test_ds, mixup_fn=mixup_fn,
                    model_overrides={"head_dim": args.head_dim}, clean_ds=clean_ds)
    print(f"best test accuracy {trainer.fit():.2f}%")


if __name__ == "__main__":
    main()
