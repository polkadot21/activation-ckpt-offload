from __future__ import annotations

from enum import IntEnum

import torch


class OffloadMode(IntEnum):
    NONE = 0
    CPU_SYNC = 1
    CPU_ASYNC = 2


class _CheckpointBlockFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, block, x, cu_seqlens, mode, offload_stream=None, profiler=None, blk_idx: int = -1
    ):
        mode = OffloadMode(mode)
        device = x.device

        if profiler is not None:
            profiler.log(f"fwd/b{blk_idx}: enter")

        with torch.no_grad():
            y = block(x, cu_seqlens)

        ctx.block = block
        ctx.cu_seqlens = cu_seqlens
        ctx.device = device
        ctx.mode = mode
        ctx.saved_gpu = None
        ctx.saved_cpu = None
        ctx.event = None
        ctx.offload_stream = None
        ctx.profiler = profiler
        ctx.blk_idx = blk_idx

        if mode == OffloadMode.NONE:
            ctx.saved_gpu = x.detach()
            if profiler is not None:
                profiler.log(f"fwd/b{blk_idx}: saved_input[gpu]")
        elif mode == OffloadMode.CPU_SYNC:
            ctx.saved_cpu = x.detach().to("cpu")
            if profiler is not None:
                profiler.log(f"fwd/b{blk_idx}: saved_input[cpu_sync]")
        elif mode == OffloadMode.CPU_ASYNC:
            x_det = x.detach()
            host_buf = torch.empty_like(x_det, device="cpu", pin_memory=True)
            if x_det.is_cuda:
                stream = (
                    offload_stream
                    if isinstance(offload_stream, torch.cuda.Stream)
                    else torch.cuda.Stream()
                )
                current = torch.cuda.current_stream(device=device)
                ev = torch.cuda.Event()
                current.record_event(ev)
                stream.wait_event(ev)
                with torch.cuda.stream(stream):
                    host_buf.copy_(x_det, non_blocking=True)
                ctx.offload_stream = stream
                ctx.event = ev
            else:
                host_buf.copy_(x_det)
            ctx.saved_cpu = host_buf
            if profiler is not None:
                profiler.log(f"fwd/b{blk_idx}: saved_input[cpu_async]")
        else:
            raise RuntimeError("Unknown offload mode")

        if profiler is not None:
            profiler.log(f"fwd/b{blk_idx}: exit")
        return y

    @staticmethod
    def backward(ctx, grad_output):
        profiler = ctx.profiler
        blk_idx = ctx.blk_idx
        if profiler is not None:
            profiler.log(f"bwd/b{blk_idx}: enter")

        block = ctx.block
        cu = ctx.cu_seqlens
        device = ctx.device
        mode = ctx.mode

        if mode == OffloadMode.NONE:
            x = ctx.saved_gpu
        else:
            if ctx.event is not None:
                ctx.event.synchronize()
            x_cpu = ctx.saved_cpu
            x = (
                x_cpu.to(device, non_blocking=(device.type == "cuda"))
                if device.type == "cuda"
                else x_cpu
            )

        if profiler is not None:
            profiler.log(f"bwd/b{blk_idx}: reloaded")

        x.requires_grad_(True)
        with torch.enable_grad():
            y = block(x, cu)

        if profiler is not None:
            profiler.log(f"bwd/b{blk_idx}: recomputed")

        torch.autograd.backward(y, grad_output)

        if profiler is not None:
            profiler.log(f"bwd/b{blk_idx}: done")

        return None, x.grad, None, None, None, None, None


def ckpt_block(block, x, cu_seqlens, mode, offload_stream=None, profiler=None, blk_idx: int = -1):
    return _CheckpointBlockFn.apply(
        block, x, cu_seqlens, int(mode), offload_stream, profiler, blk_idx
    )
