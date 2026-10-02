"""Official GSM8K task entrypoint."""

from .common import benchmark_parser, run_benchmark


def build_parser():
    return benchmark_parser("gsm8k")


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        "gsm8k", model_path=args.model_path, model_name=args.model_name,
        device=args.device, batch_size=args.batch_size, output_dir=args.output_dir,
        limit=args.limit, seed=args.seed,
    )


if __name__ == "__main__":
    main()
