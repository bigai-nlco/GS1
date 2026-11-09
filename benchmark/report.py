"""Render release figures from aggregate result files without model inference."""

import argparse
import json
from pathlib import Path


def render(directory, results=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    directory = Path(directory)
    results = Path(results) if results is not None else Path(__file__).with_name("results")
    data = json.loads((results / "public-evaluation.json").read_text())
    latency = json.loads((results / "latency.json").read_text())["measurements"]
    sizes = list(data["models"])
    colors = ["#DE8585", "#C00000", "#790000"]

    def save(fig, name):
        fig.savefig(directory / (name + ".svg"), bbox_inches="tight")
        fig.savefig(directory / (name + ".png"), bbox_inches="tight", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.9))
    scores = [data["models"][s]["public_index"]["index"] for s in sizes]
    bars = ax.bar(sizes, scores, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=5, fontweight="bold")
    ax.set(
        xlabel="GS1 model family",
        ylabel="Public index / 100",
        ylim=(0, 80),
        title="GS1: native decisions across model sizes",
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.23)
    fig.text(
        0.13,
        0.035,
        "37 public benchmarks · 100% coverage · fixed checkpoint per size",
        fontsize=9,
        color="#536171",
    )
    # The homepage model-quality.svg is the supplied report's Pareto PDF.
    save(fig, "model-size-results")

    areas = data["models"][sizes[0]]["public_index"]["areas"]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    x = np.arange(len(areas))
    for i, size in enumerate(sizes):
        values = [100 * a["skill"] for a in data["models"][size]["public_index"]["areas"]]
        bars = ax.bar(x + (i - 1) * 0.24, values, width=0.23, label="GS1-" + size, color=colors[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set(
        xticks=x,
        xticklabels=[
            "Knowledge &\nreasoning",
            "Language",
            "Retrieval &\nclassification",
            "Tools &\nautomation",
            "Arts &\ntaste",
        ],
        ylabel="Chance-corrected skill / 100",
        ylim=(0, 105),
        title="Five areas of the public Decision Index",
    )
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    save(fig, "decision-areas")

    benchmarks = data["models"][sizes[0]]["public_index"]["benchmarks"]
    ids = [k for k, v in benchmarks.items() if v["selected"] and v["in_index"]]
    matrix = np.array(
        [[100 * data["models"][s]["public_index"]["benchmarks"][k]["skill"] for s in sizes] for k in ids]
    )
    fig, ax = plt.subplots(figsize=(6.2, 11.5))
    ax.imshow(matrix, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set(
        xticks=range(3),
        xticklabels=["GS1-" + s for s in sizes],
        yticks=range(len(ids)),
        yticklabels=[benchmarks[k]["name"] for k in ids],
        title="Public benchmark breakdown",
    )
    ax.tick_params(axis="y", labelsize=9)
    for i in range(len(ids)):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > 62 else "#442222",
            )
    fig.text(
        0.12, 0.02, "Chance-corrected skill / 100; native raw metrics are in the result tables.", fontsize=8
    )
    save(fig, "benchmark-breakdown")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for j, size in enumerate(sizes):
        for i, questions in enumerate([1, 8]):
            ax = axes[i, j]
            for mode, label, color in [
                ("native_reference", "Reference", "#9ba8b6"),
                ("cuda_graph", "CUDA Graph", colors[j]),
            ]:
                rows = sorted(
                    [
                        v
                        for v in latency
                        if v["model"] == size
                        and v["questions"] == questions
                        and v["gpu_count"] == 1
                        and v["mode"] == mode
                        and v["actual_mode"] == mode
                        and v["passed"]
                    ],
                    key=lambda v: v["context"],
                )
                ax.plot(
                    [v["context"] for v in rows],
                    [v["p50_ms"] for v in rows],
                    "o-",
                    label=label,
                    color=color,
                    markersize=4,
                )
            ax.set(
                xscale="log",
                yscale="log",
                title=f"GS1-{size} · {questions} question{'s' if questions > 1 else ''}",
                xlabel="Tokens in longest branch",
                ylabel="P50 request latency (ms)",
            )
            ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "latency-context")

    fig, ax = plt.subplots(figsize=(8, 4.0))
    for i, size in enumerate(sizes):
        rows = sorted(
            [
                v
                for v in latency
                if v["model"] == size
                and v["context"] == 256
                and v["questions"] == 1
                and v["actual_mode"] == "cuda_graph"
                and v["mode"] == "cuda_graph"
                and v["passed"]
            ],
            key=lambda v: v["gpu_count"],
        )
        ax.plot(
            [v["gpu_count"] for v in rows],
            [v["requests_per_second"] for v in rows],
            "o-",
            color=colors[i],
            label="GS1-" + size,
        )
    ax.set(
        xlabel="Independent GPU replicas",
        ylabel="Measured requests / second",
        xticks=[1, 2, 4, 8],
        title="Short-window replica scaling · 256 tokens · 1 question",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    save(fig, "replica-scaling")

    fresh = results / "inference-latency.json"
    if fresh.is_file():
        runs = json.loads(fresh.read_text())["models"]
        paths = [
            ("hf_eager", "HF default", "#8C8C8C"),
            ("gs1_eager", "GS1 eager", "#DE8585"),
            ("gs1_cuda_graph", "GS1 CUDA Graph", "#C00000"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
        x = np.arange(len(runs))
        for row, count in enumerate((1, 8)):
            for col, context in enumerate((256, 1024)):
                ax = axes[row, col]
                for offset, (path, label, color) in enumerate(paths):
                    values = [
                        next(
                            v
                            for v in run["measurements"]
                            if v["context_target"] == context
                            and v["questions"] == count
                            and v["path"] == path
                        )
                        for run in runs.values()
                    ]
                    medians = [v["p50_ms"] for v in values]
                    tails = [max(0, v["p95_ms"] - v["p50_ms"]) for v in values]
                    bars = ax.bar(
                        x + (offset - 1) * 0.25,
                        medians,
                        width=0.23,
                        yerr=[np.zeros(len(runs)), tails],
                        capsize=2,
                        color=color,
                        label=label,
                        error_kw={"elinewidth": 1, "ecolor": "#6D5555"},
                    )
                    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
                ax.set(
                    xticks=x,
                    xticklabels=list(runs),
                    ylabel="Complete decision latency (ms)",
                    title=f"{context:,}-token budget · {count} question{'s' if count > 1 else ''}",
                )
                ax.set_ylim(0, ax.get_ylim()[1] * 1.17)
                ax.grid(axis="y", alpha=0.14)
                ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97))
        fig.suptitle("GS1 inference latency · current release", y=1.01, fontweight="bold")
        fig.text(
            0.5,
            -0.015,
            "RTX 4090 (48 GiB reported) · BF16 · 5 warmups + 30 calls per group\n"
            "Bars: P50; whiskers: P95 · includes compilation + probabilities; excludes load, capture and HTTP",
            ha="center",
            fontsize=9,
            color="#686060",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.92))
        save(fig, "inference-latency")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    render(args.assets, args.results)
