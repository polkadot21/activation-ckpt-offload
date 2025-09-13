from __future__ import annotations

import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from .checkpoint import OffloadMode
from .data import forward_checkpointed, make_varlen_batch
from .model import Model


def run_correctness_tests(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== Correctness tests (small sizes) ===")
    in_dim = 128
    hidden_dim = 128
    head_dim = 32
    ff_dim = 256
    num_layers = 3

    def make_model():
        torch.manual_seed(42)
        return Model(in_dim, hidden_dim, ff_dim, num_layers, head_dim).to(
            device=device, dtype=dtype
        )

    base = make_model()
    m_none = make_model()
    m_sync = make_model()
    m_async = make_model()

    t = 256
    b = 8
    x, y, cu, _ = make_varlen_batch(t, b, in_dim, device, dtype)
    loss_fn = nn.MSELoss()

    for m in (base, m_none, m_sync, m_async):
        for p in m.parameters():
            p.grad = None

    out0 = base(x, cu)
    loss0 = loss_fn(out0, y)
    loss0.backward()
    out1 = forward_checkpointed(m_none, x, cu, OffloadMode.NONE)
    loss1 = loss_fn(out1, y)
    loss1.backward()
    out2 = forward_checkpointed(m_sync, x, cu, OffloadMode.CPU_SYNC)
    loss2 = loss_fn(out2, y)
    loss2.backward()
    out3 = forward_checkpointed(m_async, x, cu, OffloadMode.CPU_ASYNC)
    loss3 = loss_fn(out3, y)
    loss3.backward()

    def same_grads(a: torch.nn.Module, b: torch.nn.Module) -> bool:
        ok = True
        for (_, p1), (__, p2) in zip(a.named_parameters(), b.named_parameters(), strict=False):
            if p1.grad is None or p2.grad is None:
                continue
            if not torch.allclose(p1.grad, p2.grad, atol=5e-3, rtol=5e-3):
                ok = False
        return ok

    ok1 = same_grads(base, m_none)
    ok2 = same_grads(base, m_sync)
    ok3 = same_grads(base, m_async)
    print(f"Grad parity vs baseline -> classic:{ok1}, sync_offload:{ok2}, async_offload:{ok3}")


def _measure_peak_mem(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device=device) / (1024**2)


def _benchmark_once(
    model: Model, x, y, cu, mode: OffloadMode | None, steps: int, device, dtype, off_stream=None
):
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize()

    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(x, cu) if mode is None else forward_checkpointed(model, x, cu, mode, off_stream)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x, cu) if mode is None else forward_checkpointed(model, x, cu, mode, off_stream)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = _measure_peak_mem(device)
    return sum(times) / len(times), peak_mb


def run_benchmarks(args) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("\n=== Running benchmarks ===")
    print(f"Device: {device}, dtype: {dtype}, flash_attn: False")

    in_dim = args.hidden_dim
    hidden_dim = args.hidden_dim
    head_dim = args.head_dim
    ff_dim = args.ff_dim
    num_layers = args.num_layers
    steps = args.steps
    total_tokens = args.total_tokens
    batch_size = args.batch_size

    x, y, cu, lens = make_varlen_batch(total_tokens, batch_size, in_dim, device, dtype)
    print(f"Total tokens: {x.size(0)} | Batch size: {batch_size} | Avg len: {sum(lens)//len(lens)}")

    def make_model():
        torch.manual_seed(123)
        return Model(in_dim, hidden_dim, ff_dim, num_layers, head_dim).to(
            device=device, dtype=dtype
        )

    results = []
    names = []
    modes = [None, OffloadMode.NONE, OffloadMode.CPU_SYNC, OffloadMode.CPU_ASYNC]
    labels = ["baseline", "ckpt_classic", "ckpt_sync_offload", "ckpt_async_offload"]

    for mode, label in zip(modes, labels, strict=False):
        model = make_model()
        off_stream = (
            torch.cuda.Stream(device=device)
            if (mode == OffloadMode.CPU_ASYNC and device.type == "cuda")
            else None
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t_ms, peak_mb = _benchmark_once(model, x, y, cu, mode, steps, device, dtype, off_stream)
        print(f"{label:>20}: time = {t_ms:8.2f} ms/iter | peak_mem = {peak_mb:8.2f} MB")
        results.append((t_ms, peak_mb))
        names.append(label)

    try:
        times = [r[0] for r in results]
        mems = [r[1] for r in results]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
        ax1.bar(names, times)
        ax1.set_title("Iteration Time (ms)")
        ax1.tick_params(axis="x", rotation=20)
        ax2.bar(names, mems)
        ax2.set_title("Peak GPU Memory (MB)")
        ax2.tick_params(axis="x", rotation=20)
        out_png = "activation_ckpt_benchmark.png"
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    except Exception as e:
        print(f"(Plot skipped) {e}")
