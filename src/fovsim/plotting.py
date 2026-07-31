"""Optional static plots for policy summaries."""

from __future__ import annotations

from pathlib import Path


def plot_frame_summary(
    rows: list[dict[str, object]],
    output: str | Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires: python -m pip install -e '.[plot]'"
        ) from exc

    frames = [int(row["display_frame"]) for row in rows]
    e3_cells = [int(row["enhancement3_cells"]) for row in rows]
    base_cells = [int(row["base_only_cells"]) for row in rows]
    cell_share = [
        100.0 * float(row["enhancement3_cell_fraction"]) for row in rows
    ]
    gaussian_share = [
        100.0 * float(row["enhancement3_gaussian_fraction"]) for row in rows
    ]
    image_share = [
        100.0 * float(row["enhancement3_image_share"]) for row in rows
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    axes[0].bar(frames, e3_cells, label="Enhancement 3", color="#2878B5")
    axes[0].bar(
        frames,
        base_cells,
        bottom=e3_cells,
        label="Base only",
        color="#B8C2CC",
    )
    axes[0].set_ylabel("Occupied cells")
    threshold = float(rows[0]["policy_threshold"])
    axes[0].set_title(
        "Visible-E3 policy: contributing_gaussian_fraction >= "
        f"{threshold:.2f}"
    )
    axes[0].legend(frameon=False, ncols=2, loc="upper right")

    axes[1].plot(
        frames,
        cell_share,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="Cells assigned E3",
        color="#2878B5",
    )
    axes[1].plot(
        frames,
        gaussian_share,
        marker="s",
        markersize=3.2,
        linewidth=1.6,
        label="Active Gaussians in E3 cells",
        color="#F28E2B",
    )
    axes[1].plot(
        frames,
        image_share,
        marker="^",
        markersize=3.5,
        linewidth=1.6,
        label="Rendered contribution covered",
        color="#2CA02C",
    )
    axes[1].set_xlabel("Display frame")
    axes[1].set_ylabel("Share (%)")
    axes[1].set_ylim(0.0, 100.0)
    if len(frames) <= 60:
        axes[1].set_xticks(frames)
    axes[1].legend(frameon=False, ncols=3, loc="lower center")
    axes[1].grid(axis="x", visible=False)

    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
