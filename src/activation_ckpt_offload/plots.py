from __future__ import annotations

import matplotlib.pyplot as plt


def plot_memory_timelines_matplotlib(traces, out_png: str = "memory_timeline.png") -> None:
    fig, axes = plt.subplots(
        len(traces), 1, figsize=(10, 4 * len(traces)), sharex=False, constrained_layout=True
    )

    if len(traces) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, traces.items(), strict=False):
        t = [s[0] for s in data]
        m = [s[1] for s in data]
        ax.plot(t, m)
        ax.set_title(f"Memory timeline: {name}")
        ax.set_ylabel("GPU MB")
        ax.grid(True, alpha=0.3)

        for tt, _, label in data:
            if "forward_done" in label:
                ax.axvline(tt, linestyle="--", color="g", alpha=0.5, label="forward_done")
            if "backward_done" in label:
                ax.axvline(tt, linestyle="--", color="r", alpha=0.5, label="backward_done")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved matplotlib memory timeline -> {out_png}")
