"""Official HellaSwag task entrypoint."""

from .common import benchmark_parser, run_benchmark


def build_parser():
    return benchmark_parser("hellaswag")


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        "hellaswag", model_path=args.model_path, model_name=args.model_name,
        device=args.device, batch_size=args.batch_size, output_dir=args.output_dir,
        limit=args.limit, seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
