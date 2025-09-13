from __future__ import annotations

import torch

from .benchmark import run_benchmarks, run_correctness_tests
from .data import run_memory_timelines
from .plots import plot_memory_timelines_matplotlib


def run_colab_demo() -> None:
    """
    Run a compact demo on Colab/T4 GPU with recommended sizes to visualize memory/time effects.
    Saves:
      - activation_ckpt_benchmark.png
      - memory_timeline.png
    """

    class Args:
        pass

    args = Args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.total_tokens = 12000
    args.batch_size = 48
    args.hidden_dim = 1024
    args.head_dim = 64
    args.ff_dim = 4096
    args.num_layers = 12
    args.steps = 3
    args.timeline_only = False
    args.plotly = False

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    run_correctness_tests(device, dtype)
    run_benchmarks(args)
    traces = run_memory_timelines(args)
    if traces:
        plot_memory_timelines_matplotlib(traces)
