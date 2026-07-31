#!/usr/bin/env python3
"""Stream an aligned FoV-policy-over-GT comparison video.

The top panel is the delivered policy model: Base Gaussians outside selected
cells plus complete E3 Gaussians inside selected cells. The bottom panel is
the DanceNet3D GT PLY. Both panels use the exact same trace camera.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_standard_ply_qoe import (
    asset_paths,
    build_ply_policy,
    camera_tensors,
    load_decisions,
    load_frame_assets,
    load_trace,
    render,
    render_inputs,
    tensor_to_image,
    validate_trace_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gsplat-library-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--hfov", type=float, default=77.0)
    parser.add_argument("--near-cm", type=float, default=1.0)
    parser.add_argument("--cell-size-m", type=float, default=0.2)
    parser.add_argument("--sh-degree", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument("--asset-frame-offset", type=int, default=1)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Optional number of evaluated frames; default renders all",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    parser.add_argument("--title-font-size", type=int, default=72)
    parser.add_argument("--detail-font-size", type=int, default=34)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--encoder-preset", default="veryfast")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "No ffmpeg executable was found and imageio-ffmpeg is absent"
        ) from exc


class RawVideoWriter:
    """One-pass raw RGB -> H.264 writer with bounded memory usage."""

    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        self.output = output
        self.log_path = output.with_suffix(".ffmpeg.log")
        self._log: BinaryIO = self.log_path.open("wb")
        command = [
            find_ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._log.close()
        if return_code != 0:
            details = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(
                f"FFmpeg failed with exit code {return_code}: {details}"
            )


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError as exc:
        raise RuntimeError("No scalable bold font is available") from exc


def add_label(
    image: Image.Image,
    title: str,
    detail: str,
    title_font: ImageFont.ImageFont,
    detail_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    x, y = 34, 28
    draw.text(
        (x, y),
        title,
        font=title_font,
        fill=(255, 255, 255),
        stroke_width=5,
        stroke_fill=(0, 0, 0),
    )
    title_bounds = draw.textbbox(
        (x, y), title, font=title_font, stroke_width=5
    )
    draw.text(
        (x, title_bounds[3] + 14),
        detail,
        font=detail_font,
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )


def compose_frame(
    policy: torch.Tensor,
    gt: torch.Tensor,
    index: int,
    total: int,
    decision: dict[str, Any],
    trace_name: str,
    title_font: ImageFont.ImageFont,
    detail_font: ImageFont.ImageFont,
) -> np.ndarray:
    policy_image = tensor_to_image(policy)
    gt_image = tensor_to_image(gt)
    detail = (
        f"{trace_name}  |  Frame {index + 1}/{total}  |  "
        f"GSV {decision['gsv_frame']:03d}  |  "
        f"selected E3 cells: {len(decision['selected_cells'])}"
    )
    add_label(
        policy_image,
        "FoV Policy (Base + Selected E3)",
        detail,
        title_font,
        detail_font,
    )
    add_label(
        gt_image,
        "DanceNet3D GT (GSV)",
        detail,
        title_font,
        detail_font,
    )
    canvas = Image.new(
        "RGB",
        (policy_image.width, policy_image.height + gt_image.height),
        (0, 0, 0),
    )
    canvas.paste(policy_image, (0, 0))
    canvas.paste(gt_image, (0, policy_image.height))
    return np.ascontiguousarray(np.asarray(canvas), dtype=np.uint8)


def validate_args(args: argparse.Namespace) -> None:
    for role, path in (
        ("trace", args.trace),
        ("decisions", args.decisions),
        ("model root", args.model_root),
        ("GT root", args.gt_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {role}: {path}")
    if args.first_frame <= 0:
        raise ValueError("--first-frame must be positive")
    if args.frame_count is not None and args.frame_count <= 0:
        raise ValueError("--frame-count must be positive")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0.0:
        raise ValueError("Image dimensions and FPS must be positive")
    if args.cell_size_m <= 0.0 or args.prefetch_workers <= 0:
        raise ValueError("Cell size and prefetch workers must be positive")
    if args.title_font_size <= 0 or args.detail_font_size <= 0:
        raise ValueError("Font sizes must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be within [0, 51]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.gsplat_library_path.resolve()))
    from gsplat.cuda._wrapper import spherical_harmonics
    from gsplat.exporter import load_ply_to_splats
    from gsplat.rendering import rasterization

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    trace_rows = load_trace(args.trace)
    all_decisions = load_decisions(args.decisions)
    first_index = args.first_frame - 1
    end_index = (
        first_index + args.frame_count
        if args.frame_count is not None
        else None
    )
    decisions = all_decisions[first_index:end_index]
    if not decisions:
        raise ValueError("Selected decision interval is empty")
    if args.frame_count is not None and len(decisions) != args.frame_count:
        raise ValueError(
            f"Requested {args.frame_count} frames but only "
            f"{len(decisions)} are available"
        )

    specs: list[dict[str, Any]] = []
    for decision in decisions:
        source_row = decision["trace_source_row"]
        trace = trace_rows.get(source_row)
        if trace is None:
            raise ValueError(f"Unknown trace source row {source_row}")
        validate_trace_reference(trace, decision)
        asset_id = decision["gsv_frame"] + args.asset_frame_offset
        specs.append(
            {
                "trace": trace,
                "asset_id": asset_id,
                "paths": asset_paths(args.model_root, args.gt_root, asset_id),
            }
        )

    trace_name = args.trace.stem
    output_path = args.output_dir / "policy_top_gt_bottom_30fps.mp4"
    writer = RawVideoWriter(
        output_path,
        args.width,
        args.height * 2,
        args.fps,
        args.crf,
        args.encoder_preset,
    )
    title_font = load_font(args.title_font_size)
    detail_font = load_font(args.detail_font_size)
    started = time.perf_counter()
    timings = {
        "asset_wait_wall_seconds": 0.0,
        "asset_load_cpu_work_seconds": 0.0,
        "policy_build_seconds": 0.0,
        "render_seconds": 0.0,
        "compose_encode_seconds": 0.0,
    }
    loader_threads: set[str] = set()
    records: list[dict[str, Any]] = []
    prefetch_window = args.prefetch_workers + 1
    futures: dict[int, Future[dict[str, Any]]] = {}

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        with ThreadPoolExecutor(
            max_workers=args.prefetch_workers,
            thread_name_prefix="video-ply",
        ) as pool:
            for index in range(min(prefetch_window, len(specs))):
                futures[index] = pool.submit(
                    load_frame_assets,
                    specs[index]["paths"],
                    load_ply_to_splats,
                )

            with torch.inference_mode():
                for index, (decision, spec) in enumerate(
                    zip(decisions, specs, strict=True)
                ):
                    wait_started = time.perf_counter()
                    assets = futures.pop(index).result()
                    timings["asset_wait_wall_seconds"] += (
                        time.perf_counter() - wait_started
                    )
                    timings["asset_load_cpu_work_seconds"] += float(
                        assets["load_seconds"]
                    )
                    loader_threads.add(str(assets["loader_thread"]))
                    next_index = index + prefetch_window
                    if next_index < len(specs):
                        futures[next_index] = pool.submit(
                            load_frame_assets,
                            specs[next_index]["paths"],
                            load_ply_to_splats,
                        )

                    policy_started = time.perf_counter()
                    policy, policy_stats = build_ply_policy(
                        assets["base"],
                        assets["e3"],
                        decision["selected_cells"],
                        args.cell_size_m,
                    )
                    timings["policy_build_seconds"] += (
                        time.perf_counter() - policy_started
                    )

                    render_started = time.perf_counter()
                    viewmat, intrinsics = camera_tensors(
                        spec["trace"],
                        args.width,
                        args.height,
                        args.hfov,
                        device,
                    )
                    policy_inputs = render_inputs(
                        policy,
                        spec["trace"],
                        device,
                        args.sh_degree,
                        spherical_harmonics,
                    )
                    gt_inputs = render_inputs(
                        assets["gt"],
                        spec["trace"],
                        device,
                        args.sh_degree,
                        spherical_harmonics,
                    )
                    policy_image, _ = render(
                        rasterization,
                        policy_inputs,
                        viewmat,
                        intrinsics,
                        args.width,
                        args.height,
                        args.near_cm,
                    )
                    gt_image, _ = render(
                        rasterization,
                        gt_inputs,
                        viewmat,
                        intrinsics,
                        args.width,
                        args.height,
                        args.near_cm,
                    )
                    synchronize()
                    timings["render_seconds"] += (
                        time.perf_counter() - render_started
                    )

                    encode_started = time.perf_counter()
                    composite = compose_frame(
                        policy_image,
                        gt_image,
                        index,
                        len(decisions),
                        decision,
                        trace_name,
                        title_font,
                        detail_font,
                    )
                    writer.write(composite)
                    timings["compose_encode_seconds"] += (
                        time.perf_counter() - encode_started
                    )
                    if index in (0, len(decisions) // 2, len(decisions) - 1):
                        Image.fromarray(composite).save(
                            args.output_dir / f"qa_frame_{index + 1:04d}.png"
                        )

                    records.append(
                        {
                            "video_frame": index,
                            "output_frame": decision["output_frame"],
                            "output_time_s": decision["output_time_s"],
                            "trace_source_row": decision["trace_source_row"],
                            "trace_timestamp_s": spec["trace"]["timestamp_s"],
                            "gsv_frame": decision["gsv_frame"],
                            "asset_frame_id": spec["asset_id"],
                            "selected_cell_count": len(
                                decision["selected_cells"]
                            ),
                            **policy_stats,
                        }
                    )
                    if index == 0 or (index + 1) % 100 == 0:
                        elapsed = time.perf_counter() - started
                        print(
                            f"Rendered {index + 1}/{len(decisions)} | "
                            f"{(index + 1) / elapsed:.2f} fps | "
                            f"GSV {decision['gsv_frame']}",
                            flush=True,
                        )
                    del (
                        assets,
                        policy,
                        policy_inputs,
                        gt_inputs,
                        policy_image,
                        gt_image,
                        composite,
                    )
        writer.close()
    except BaseException:
        if writer._process.poll() is None:
            writer._process.terminate()
            writer._process.wait()
        writer._log.close()
        raise

    elapsed = time.perf_counter() - started
    metadata = {
        "pipeline_status": "PASS",
        "policy_definition": (
            "Base Gaussians outside cells selected by the decision CSV plus "
            "full E3 Gaussians inside selected cells; "
            f"cell_size_m={args.cell_size_m}"
        ),
        "layout": {
            "top": "FoV policy (Base + selected E3)",
            "bottom": "DanceNet3D GT (GSV)",
            "title_font_size": args.title_font_size,
            "detail_font_size": args.detail_font_size,
        },
        "inputs": {
            "trace": str(args.trace.resolve()),
            "decisions": str(args.decisions.resolve()),
            "model_root": str(args.model_root.resolve()),
            "gt_root": str(args.gt_root.resolve()),
        },
        "video": {
            "path": str(output_path.resolve()),
            "codec": "H.264/libx264",
            "fps": args.fps,
            "frame_count": len(decisions),
            "duration_seconds": len(decisions) / args.fps,
            "width": args.width,
            "height": args.height * 2,
            "crf": args.crf,
            "preset": args.encoder_preset,
        },
        "projection": {
            "panel_width": args.width,
            "panel_height": args.height,
            "hfov_degrees": args.hfov,
            "near_cm": args.near_cm,
        },
        "parallelism": {
            "prefetch_workers": args.prefetch_workers,
            "loader_threads": sorted(loader_threads),
            "streaming_encode": True,
            "cuda_rendering": device.type == "cuda",
        },
        "performance": {
            **timings,
            "total_wall_seconds": elapsed,
            "throughput_frames_per_second": len(decisions) / elapsed,
        },
        "rendering": {
            "renderer": "gsplat CUDA",
            "sh_degree": args.sh_degree,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "frames": records,
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote video: {output_path}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(
        f"PASS: {len(decisions)} frames in {elapsed:.2f}s "
        f"({len(decisions) / elapsed:.2f} fps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
