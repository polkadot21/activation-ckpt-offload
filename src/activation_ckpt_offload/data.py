from __future__ import annotations

import time

import torch
from torch import nn

from .checkpoint import OffloadMode, ckpt_block
from .model import Model


class GpuMemProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.t0 = time.perf_counter()
        self.samples: list[tuple[float, float, str]] = []

    def _now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def _mem_mb(self) -> float:
        if self.device.type != "cuda":
            return float("nan")
        return torch.cuda.memory_allocated(device=self.device) / (1024**2)

    def log(self, label: str) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        self.samples.append((self._now_ms(), self._mem_mb(), label))


def make_varlen_batch(
    total_tokens: int, batch_size: int, dim: int, device: torch.device, dtype: torch.dtype
):
    rng = torch.Generator(device="cpu")
    lengths = torch.randint(
        low=max(4, total_tokens // (batch_size * 2)),
        high=max(5, 2 * total_tokens // batch_size),
        size=(batch_size,),
        generator=rng,
    ).tolist()
    s = sum(lengths)
    lengths = [max(1, length * total_tokens // s) for length in lengths]
    diff = total_tokens - sum(lengths)
    lengths[-1] += diff

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    cu = torch.tensor(cu, device=device, dtype=torch.int32)
    t = cu[-1].item()
    x = torch.randn(t, dim, device=device, dtype=dtype)
    y = torch.randn(t, dim, device=device, dtype=dtype)
    return x, y, cu, lengths


def forward_checkpointed(
    model: Model,
    x: torch.Tensor,
    cu: torch.Tensor,
    mode: OffloadMode,
    offload_stream=None,
    profiler: GpuMemProfiler | None = None,
) -> torch.Tensor:
    x = model.in_layer(x)
    for i, blk in enumerate(model.blocks):
        x = ckpt_block(blk, x, cu, mode, offload_stream, profiler, i)
    return model.out_layer(x)


def run_memory_timelines(args):
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    if device.type != "cuda":
        print("Memory timeline requires CUDA; skipping.")
        return {}

    dtype = torch.float16
    in_dim = args.hidden_dim
    hidden_dim = args.hidden_dim
    head_dim = args.head_dim
    ff_dim = args.ff_dim
    num_layers = args.num_layers
    total_tokens = args.total_tokens
    batch_size = args.batch_size

    x, y, cu, _ = make_varlen_batch(total_tokens, batch_size, in_dim, device, dtype)

    def make_model():
        torch.manual_seed(123)
        return Model(in_dim, hidden_dim, ff_dim, num_layers, head_dim).to(
            device=device, dtype=dtype
        )

    modes = [
        (OffloadMode.NONE, "ckpt_classic"),
        (OffloadMode.CPU_SYNC, "ckpt_sync_offload"),
        (OffloadMode.CPU_ASYNC, "ckpt_async_offload"),
    ]
    traces: dict[str, list[tuple[float, float, str]]] = {}

    for mode, name in modes:
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prof = GpuMemProfiler(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        prof.log(f"{name}: step_start")
        opt.zero_grad(set_to_none=True)
        out = forward_checkpointed(model, x, cu, mode, offload_stream=None, profiler=prof)
        prof.log(f"{name}: forward_done")
        loss = nn.MSELoss()(out, y)
        prof.log(f"{name}: loss_ready")
        loss.backward()
        prof.log(f"{name}: backward_done")
        opt.step()
        prof.log(f"{name}: step_end")

        traces[name] = prof.samples

    return traces
