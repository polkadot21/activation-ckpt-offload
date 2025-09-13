from __future__ import annotations

import argparse

import torch

from .benchmark import run_benchmarks, run_correctness_tests
from .data import run_memory_timelines
from .plots import plot_memory_timelines_matplotlib


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Activation Checkpointing with CPU Offload")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--total_tokens", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--ff_dim", type=int, default=2048)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument(
        "--timeline-only",
        action="store_true",
        help="Only generate timeline plots (skip benchmarks).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available -> using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if not args.timeline_only:
        run_correctness_tests(device, dtype)
        run_benchmarks(args)

    traces = run_memory_timelines(args)
    if traces:
        plot_memory_timelines_matplotlib(traces)
