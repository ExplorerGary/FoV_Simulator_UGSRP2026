"""Direct linear prediction from recent 6DoF to future cell visibility."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from .io import write_csv_atomic, write_json_atomic
from .trace import TraceRow, load_trace

if TYPE_CHECKING:
    import numpy as np


VISIBILITY_REQUIRED_COLUMNS = frozenset(
    {
        "output_frame",
        "trace_timestamp_s",
        "cell_id",
        "contributing_gaussian_fraction",
    }
)


@dataclass(slots=True)
class SampledTrace:
    name: str
    path: Path
    duration_s: float
    source_rows: int
    effective_fps: float
    times_s: "np.ndarray"
    dof: "np.ndarray"


@dataclass(slots=True)
class VisibilityFrames:
    output_frames: "np.ndarray"
    times_s: "np.ndarray"
    values_by_frame: list[dict[str, float]]


@dataclass(slots=True)
class Examples:
    features: "np.ndarray"
    targets: "np.ndarray"
    current_visibility: "np.ndarray"
    actual_horizons_s: "np.ndarray"
    current_indices: "np.ndarray"
    target_indices: "np.ndarray"


@dataclass(slots=True)
class StandardizedRidge:
    feature_mean: "np.ndarray"
    feature_scale: "np.ndarray"
    target_mean: "np.ndarray"
    target_scale: "np.ndarray"
    coefficients: "np.ndarray"
    alpha: float

    @classmethod
    def fit(
        cls,
        features: "np.ndarray",
        targets: "np.ndarray",
        alpha: float,
    ) -> "StandardizedRidge":
        import numpy as np

        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("Linear-regression inputs and targets must be 2D")
        if len(features) != len(targets) or len(features) == 0:
            raise ValueError("Linear regression requires matching non-empty data")
        if alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")

        feature_mean = features.mean(axis=0)
        feature_scale = features.std(axis=0)
        feature_scale[feature_scale < 1e-12] = 1.0
        target_mean = targets.mean(axis=0)
        target_scale = targets.std(axis=0)
        target_scale[target_scale < 1e-12] = 1.0
        x = (features - feature_mean) / feature_scale
        y = (targets - target_mean) / target_scale
        if alpha == 0.0:
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            gram = x.T @ x
            gram.flat[:: gram.shape[0] + 1] += alpha
            coefficients = np.linalg.solve(gram, x.T @ y)
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            coefficients=coefficients,
            alpha=alpha,
        )

    def predict(self, features: "np.ndarray") -> "np.ndarray":
        x = (features - self.feature_mean) / self.feature_scale
        return (x @ self.coefficients) * self.target_scale + self.target_mean

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size + self.target_mean.size)

    def save(
        self,
        path: Path,
        *,
        cell_ids: Sequence[str],
        decision_threshold: float,
        target_threshold: float,
        training_target_mode: str,
        feature_mode: str = "raw_history",
    ) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            coefficients=self.coefficients,
            alpha=np.asarray([self.alpha], dtype=np.float64),
            cell_ids=np.asarray(cell_ids),
            input_contract=np.asarray(["6dof_history_only"]),
            target_contract=np.asarray(
                ["future_contributing_gaussian_fraction"]
            ),
            decision_threshold=np.asarray([decision_threshold], dtype=np.float64),
            target_threshold=np.asarray([target_threshold], dtype=np.float64),
            training_target_mode=np.asarray([training_target_mode]),
            feature_mode=np.asarray([feature_mode]),
        )


def _matching_rows(rows: Iterable[TraceRow], sequence: str) -> list[TraceRow]:
    return [row for row in rows if row.file_name == sequence]


def _load_sampled_trace(path: Path, sequence: str, fps: float) -> SampledTrace | None:
    import numpy as np

    rows, _ = load_trace(path)
    selected = _matching_rows(rows, sequence)
    if not selected:
        return None
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    timestamps = np.asarray([row.timestamp_s for row in selected], dtype=np.float64)
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0.0:
        raise ValueError(f"Trace {path} has no positive-duration {sequence} data")
    sample_count = int(math.floor(duration * fps + 1e-9)) + 1
    times = timestamps[0] + np.arange(sample_count, dtype=np.float64) / fps
    source_dof = np.asarray(
        [row.location_cm + row.rotation_rpy_degrees for row in selected],
        dtype=np.float64,
    )
    source_dof[:, 3:] = np.rad2deg(
        np.unwrap(np.deg2rad(source_dof[:, 3:]), axis=0)
    )
    sampled = np.column_stack(
        [np.interp(times, timestamps, source_dof[:, column]) for column in range(6)]
    )
    return SampledTrace(
        name=path.stem,
        path=path,
        duration_s=duration,
        source_rows=len(selected),
        effective_fps=(len(selected) - 1) / duration,
        times_s=times,
        dof=sampled,
    )


def discover_traces(
    trace_dir: str | Path,
    *,
    sequence: str,
    fps: float,
) -> list[SampledTrace]:
    directory = Path(trace_dir)
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV traces found in {directory}")
    traces: list[SampledTrace] = []
    for path in paths:
        trace = _load_sampled_trace(path, sequence, fps)
        if trace is not None:
            traces.append(trace)
    if not traces:
        raise ValueError(f"No traces contain FileName={sequence!r}")
    return traces


def load_visibility(path: str | Path) -> VisibilityFrames:
    import numpy as np

    visibility_path = Path(path)
    frames: dict[int, tuple[float, dict[str, float]]] = {}
    with visibility_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(
            VISIBILITY_REQUIRED_COLUMNS - set(reader.fieldnames or ())
        )
        if missing:
            raise ValueError(
                f"Visibility {visibility_path} is missing columns: {missing}"
            )
        for source_row, raw in enumerate(reader, start=2):
            try:
                frame = int(raw["output_frame"])
                timestamp = float(raw["trace_timestamp_s"])
                fraction = float(raw["contributing_gaussian_fraction"])
                cell_id = raw["cell_id"]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid visibility value at {visibility_path}:{source_row}"
                ) from exc
            if not math.isfinite(timestamp) or not math.isfinite(fraction):
                raise ValueError(
                    f"Non-finite visibility value at {visibility_path}:{source_row}"
                )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"Visibility fraction outside [0, 1] at "
                    f"{visibility_path}:{source_row}"
                )
            if frame not in frames:
                frames[frame] = (timestamp, {})
            existing_time, values = frames[frame]
            if abs(existing_time - timestamp) > 1e-6:
                raise ValueError(
                    f"Frame {frame} has inconsistent timestamps in {visibility_path}"
                )
            if cell_id in values:
                raise ValueError(
                    f"Duplicate cell {cell_id!r} in frame {frame} of {visibility_path}"
                )
            values[cell_id] = fraction
    if not frames:
        raise ValueError(f"Visibility {visibility_path} contains no rows")
    frame_numbers = np.asarray(sorted(frames), dtype=np.int64)
    if len(frame_numbers) > 1 and np.any(np.diff(frame_numbers) != 1):
        raise ValueError(
            f"Visibility output frames are not consecutive in {visibility_path}"
        )
    ordered = [frames[int(frame)] for frame in frame_numbers]
    times = np.asarray([item[0] for item in ordered], dtype=np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Visibility timestamps are not increasing in {visibility_path}")
    return VisibilityFrames(
        output_frames=frame_numbers,
        times_s=times,
        values_by_frame=[item[1] for item in ordered],
    )


def _visibility_matrix(
    frames: VisibilityFrames,
    cell_index: dict[str, int],
) -> "np.ndarray":
    import numpy as np

    values = np.zeros((len(frames.values_by_frame), len(cell_index)), dtype=np.float64)
    for frame_index, frame in enumerate(frames.values_by_frame):
        for cell_id, fraction in frame.items():
            values[frame_index, cell_index[cell_id]] = fraction
    return values


def _interpolate_dof(trace: SampledTrace, times_s: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if times_s[0] < trace.times_s[0] - 1e-6 or times_s[-1] > trace.times_s[-1] + 1e-6:
        raise ValueError(f"Visibility times fall outside trace {trace.name}")
    return np.column_stack(
        [
            np.interp(times_s, trace.times_s, trace.dof[:, column])
            for column in range(6)
        ]
    )


def _examples(
    trace: SampledTrace,
    visibility: VisibilityFrames,
    cell_index: dict[str, int],
    *,
    fps: float,
    history_steps: int,
    horizon_s: float,
    feature_mode: str = "raw_history",
) -> Examples:
    import numpy as np

    visibility_values = _visibility_matrix(visibility, cell_index)
    history_offsets = np.arange(history_steps, -1, -1, dtype=np.float64) / fps
    histories: list[np.ndarray] = []
    current_indices: list[int] = []
    target_indices: list[int] = []
    max_target_error_s = 1.1 / fps
    for index in range(len(visibility_values)):
        history_times = visibility.times_s[index] - history_offsets
        if history_times[0] < trace.times_s[0] - 1e-6:
            continue
        desired_time = visibility.times_s[index] + horizon_s
        right = int(np.searchsorted(visibility.times_s, desired_time))
        candidates = [
            candidate
            for candidate in (right - 1, right)
            if index < candidate < len(visibility.times_s)
        ]
        if not candidates:
            continue
        target_index = min(
            candidates,
            key=lambda candidate: abs(
                float(visibility.times_s[candidate]) - desired_time
            ),
        )
        if abs(float(visibility.times_s[target_index]) - desired_time) > max_target_error_s:
            continue
        histories.append(_interpolate_dof(trace, history_times))
        current_indices.append(index)
        target_indices.append(target_index)
    if not histories:
        raise ValueError(f"No aligned visibility examples remain for {trace.name}")
    current = np.asarray(current_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    history_array = np.stack(histories)
    return Examples(
        features=_history_features(
            history_array,
            fps=fps,
            horizon_s=horizon_s,
            feature_mode=feature_mode,
        ),
        targets=visibility_values[target],
        current_visibility=visibility_values[current],
        actual_horizons_s=visibility.times_s[target] - visibility.times_s[current],
        current_indices=current,
        target_indices=target,
    )


def _history_features(
    histories: "np.ndarray",
    *,
    fps: float,
    horizon_s: float,
    feature_mode: str,
) -> "np.ndarray":
    """Build fixed features while keeping Ridge as the only learned model."""
    import numpy as np

    if feature_mode == "raw_history":
        return histories.reshape(len(histories), -1)
    if feature_mode != "motion_quadratic":
        raise ValueError(
            "feature_mode must be 'raw_history' or 'motion_quadratic'"
        )
    if histories.shape[1] < 2:
        raise ValueError("motion_quadratic requires at least two history samples")

    current = histories[:, -1, :]
    relative_history = histories[:, :-1, :] - current[:, None, :]
    history_duration = (histories.shape[1] - 1) / fps
    long_velocity = (current - histories[:, 0, :]) / history_duration
    short_steps = min(3, histories.shape[1] - 1)
    short_duration = short_steps / fps
    short_velocity = (
        current - histories[:, -1 - short_steps, :]
    ) / short_duration
    if histories.shape[1] >= 2 * short_steps + 1:
        earlier_velocity = (
            histories[:, -1 - short_steps, :]
            - histories[:, -1 - 2 * short_steps, :]
        ) / short_duration
        acceleration = (short_velocity - earlier_velocity) / short_duration
    else:
        acceleration = np.zeros_like(short_velocity)
    extrapolated = (
        current
        + short_velocity * horizon_s
        + 0.5 * acceleration * horizon_s * horizon_s
    )
    quadratic_columns = [
        extrapolated[:, left] * extrapolated[:, right]
        for left in range(6)
        for right in range(left, 6)
    ]
    quadratic = np.column_stack(quadratic_columns)
    return np.concatenate(
        [
            current,
            relative_history.reshape(len(histories), -1),
            long_velocity,
            short_velocity,
            acceleration,
            extrapolated,
            quadratic,
            current * short_velocity,
        ],
        axis=1,
    )


def _regression_metrics(
    prediction: "np.ndarray",
    target: "np.ndarray",
) -> dict[str, float]:
    import numpy as np

    clipped = np.clip(prediction, 0.0, 1.0)
    return {"mse": float(np.mean((clipped - target) ** 2))}


def _classification_metrics(
    prediction: "np.ndarray",
    target: "np.ndarray",
    prediction_threshold: float,
    target_threshold: float,
) -> dict[str, float | int]:
    import numpy as np

    predicted = np.clip(prediction, 0.0, 1.0) >= prediction_threshold
    expected = target >= target_threshold
    true_positive = int(np.count_nonzero(predicted & expected))
    true_negative = int(np.count_nonzero(~predicted & ~expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    total = true_positive + true_negative + false_positive + false_negative
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 4.0
    f2 = (
        (1.0 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if precision + recall
        else 0.0
    )
    frame_count = int(prediction.shape[0])
    return {
        "prediction_threshold": prediction_threshold,
        "target_threshold": target_threshold,
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "f2": f2,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "positive_fraction": float(np.mean(expected)),
        "target_visible_cells_per_frame": (
            true_positive + false_negative
        ) / frame_count,
        "predicted_cells_per_frame": (
            true_positive + false_positive
        ) / frame_count,
        "missed_visible_cells_per_frame": false_negative / frame_count,
        "extra_cells_per_frame": false_positive / frame_count,
    }


def _metrics(
    prediction: "np.ndarray",
    target: "np.ndarray",
    prediction_threshold: float,
    target_threshold: float,
) -> dict[str, object]:
    return {
        **_regression_metrics(prediction, target),
        "classification": _classification_metrics(
            prediction,
            target,
            prediction_threshold,
            target_threshold,
        ),
    }


def _select_decision_threshold(
    prediction: "np.ndarray",
    target: "np.ndarray",
    *,
    target_threshold: float,
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, dict[str, float | int]]:
    import numpy as np

    candidates = np.linspace(minimum, maximum, steps)
    scored = [
        (
            float(candidate),
            _classification_metrics(
                prediction,
                target,
                float(candidate),
                target_threshold,
            ),
        )
        for candidate in candidates
    ]
    selected, metrics = max(
        scored,
        key=lambda item: (
            float(item[1]["f2"]),
            float(item[1]["recall"]),
            float(item[1]["f1"]),
            -item[0],
        ),
    )
    return selected, metrics


def _concat(examples: Sequence[Examples], field: str) -> "np.ndarray":
    import numpy as np

    return np.concatenate([getattr(example, field) for example in examples], axis=0)


def _split_names(
    names: Sequence[str], test_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    shuffled = list(sorted(names))
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    if test_count >= len(shuffled):
        raise ValueError("At least one training trace is required")
    return set(shuffled[test_count:]), set(shuffled[:test_count])


def run_linear_prediction(
    *,
    trace_dir: str | Path,
    visibility_dir: str | Path,
    output_dir: str | Path,
    sequence: str = "BiancaGolden_CircleTurns",
    fps: float = 30.0,
    history_ms: int = 500,
    horizons_ms: Sequence[int] = (100, 200, 500),
    visibility_threshold: float = 0.5,
    decision_threshold_min: float = 0.01,
    decision_threshold_max: float = 0.5,
    decision_threshold_steps: int = 50,
    target_mode: str = "fraction",
    test_fraction: float = 0.2,
    seed: int = 20260731,
    ridge_alpha: float = 1.0,
    expected_traces: int | None = None,
    feature_mode: str = "raw_history",
) -> dict[str, object]:
    """Fit direct DoF-history-to-cell-visibility regressors."""
    import numpy as np

    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_threshold must be within [0, 1]")
    if not 0.0 <= decision_threshold_min < decision_threshold_max <= 1.0:
        raise ValueError("Invalid decision-threshold search range")
    if decision_threshold_steps < 2:
        raise ValueError("decision_threshold_steps must be at least 2")
    if target_mode not in {"fraction", "binary"}:
        raise ValueError("target_mode must be 'fraction' or 'binary'")
    if feature_mode not in {"raw_history", "motion_quadratic"}:
        raise ValueError("Invalid feature_mode")
    traces = discover_traces(trace_dir, sequence=sequence, fps=fps)
    if expected_traces is not None and len(traces) != expected_traces:
        raise ValueError(
            f"Expected {expected_traces} {sequence} traces, found {len(traces)}"
        )
    if len(traces) < 2:
        raise ValueError("Prediction evaluation requires at least two traces")
    if history_ms <= 0:
        raise ValueError("history_ms must be positive")
    if not horizons_ms or any(horizon <= 0 for horizon in horizons_ms):
        raise ValueError("horizons_ms must contain positive values")

    history_steps = max(1, int(round(history_ms * fps / 1000.0)))
    horizon_steps = {
        int(horizon): max(1, int(round(horizon * fps / 1000.0)))
        for horizon in horizons_ms
    }
    train_names, test_names = _split_names(
        [trace.name for trace in traces], test_fraction, seed
    )
    threshold_fit_names, calibration_names = _split_names(
        sorted(train_names), 0.25, seed + 1
    )

    visibility_by_name: dict[str, VisibilityFrames] = {}
    all_cells: set[str] = set()
    visibility_root = Path(visibility_dir)
    for trace in traces:
        path = visibility_root / f"{trace.name}.csv"
        if not path.exists():
            raise ValueError(f"Missing visibility CSV for {trace.name}: {path}")
        frames = load_visibility(path)
        visibility_by_name[trace.name] = frames
        for frame in frames.values_by_frame:
            all_cells.update(frame)
    cell_ids = sorted(all_cells)
    if not cell_ids:
        raise ValueError("Visibility inputs contain no cell IDs")
    cell_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    horizon_results: dict[str, object] = {}
    for horizon_ms, steps in horizon_steps.items():
        examples_by_name = {
            trace.name: _examples(
                trace,
                visibility_by_name[trace.name],
                cell_index,
                fps=fps,
                history_steps=history_steps,
                horizon_s=horizon_ms / 1000.0,
                feature_mode=feature_mode,
            )
            for trace in traces
        }
        training = [examples_by_name[name] for name in sorted(train_names)]
        testing = [examples_by_name[name] for name in sorted(test_names)]
        threshold_fitting = [
            examples_by_name[name] for name in sorted(threshold_fit_names)
        ]
        calibration = [
            examples_by_name[name] for name in sorted(calibration_names)
        ]
        threshold_model = StandardizedRidge.fit(
            _concat(threshold_fitting, "features"),
            (
                _concat(threshold_fitting, "targets")
                if target_mode == "fraction"
                else (
                    _concat(threshold_fitting, "targets")
                    >= visibility_threshold
                ).astype(np.float64)
            ),
            ridge_alpha,
        )
        calibration_y = _concat(calibration, "targets")
        decision_threshold, calibration_metrics = _select_decision_threshold(
            threshold_model.predict(_concat(calibration, "features")),
            calibration_y,
            target_threshold=visibility_threshold,
            minimum=decision_threshold_min,
            maximum=decision_threshold_max,
            steps=decision_threshold_steps,
        )
        train_x = _concat(training, "features")
        train_fraction_targets = _concat(training, "targets")
        train_y = (
            train_fraction_targets
            if target_mode == "fraction"
            else (train_fraction_targets >= visibility_threshold).astype(np.float64)
        )
        model = StandardizedRidge.fit(train_x, train_y, ridge_alpha)
        model.save(
            output / f"visibility_model_{horizon_ms}ms.npz",
            cell_ids=cell_ids,
            decision_threshold=decision_threshold,
            target_threshold=visibility_threshold,
            training_target_mode=target_mode,
            feature_mode=feature_mode,
        )

        test_x = _concat(testing, "features")
        test_y = _concat(testing, "targets")
        current = _concat(testing, "current_visibility")
        actual_horizons = _concat(testing, "actual_horizons_s")
        result = _metrics(
            model.predict(test_x),
            test_y,
            decision_threshold,
            visibility_threshold,
        )
        baseline = _metrics(
            current,
            test_y,
            visibility_threshold,
            visibility_threshold,
        )
        per_trace: dict[str, object] = {}
        for name in sorted(test_names):
            example = examples_by_name[name]
            trace_result = _metrics(
                model.predict(example.features),
                example.targets,
                decision_threshold,
                visibility_threshold,
            )
            trace_baseline = _metrics(
                example.current_visibility,
                example.targets,
                visibility_threshold,
                visibility_threshold,
            )
            per_trace[name] = {
                "samples": len(example.features),
                "visibility": trace_result,
                "persistence": trace_baseline,
            }
            classification = trace_result["classification"]
            baseline_classification = trace_baseline["classification"]
            assert isinstance(classification, dict)
            assert isinstance(baseline_classification, dict)
            metrics_rows.append(
                {
                    "horizon_ms": horizon_ms,
                    "trace": name,
                    "samples": len(example.features),
                    "decision_threshold": decision_threshold,
                    "mse": trace_result["mse"],
                    "persistence_mse": trace_baseline["mse"],
                    "accuracy": classification["accuracy"],
                    "precision": classification["precision"],
                    "recall": classification["recall"],
                    "f1": classification["f1"],
                    "f2": classification["f2"],
                    "balanced_accuracy": classification["balanced_accuracy"],
                    "target_visible_cells_per_frame": classification[
                        "target_visible_cells_per_frame"
                    ],
                    "predicted_cells_per_frame": classification[
                        "predicted_cells_per_frame"
                    ],
                    "missed_visible_cells_per_frame": classification[
                        "missed_visible_cells_per_frame"
                    ],
                    "extra_cells_per_frame": classification[
                        "extra_cells_per_frame"
                    ],
                    "persistence_accuracy": baseline_classification["accuracy"],
                    "persistence_f1": baseline_classification["f1"],
                    "persistence_f2": baseline_classification["f2"],
                    "persistence_missed_cells_per_frame": baseline_classification[
                        "missed_visible_cells_per_frame"
                    ],
                    "persistence_extra_cells_per_frame": baseline_classification[
                        "extra_cells_per_frame"
                    ],
                }
            )
        horizon_results[f"{horizon_ms}ms"] = {
            "mean_actual_horizon_ms": float(1000.0 * np.mean(actual_horizons)),
            "min_actual_horizon_ms": float(1000.0 * np.min(actual_horizons)),
            "max_actual_horizon_ms": float(1000.0 * np.max(actual_horizons)),
            "horizon_steps": steps,
            "input_dimension": int(train_x.shape[1]),
            "output_cells": len(cell_ids),
            "training_samples": int(len(train_x)),
            "test_samples": int(len(test_x)),
            "model_parameters": model.parameter_count,
            "threshold_calibration": {
                "objective": "maximum_f2",
                "fit_traces": sorted(threshold_fit_names),
                "calibration_traces": sorted(calibration_names),
                "selected_decision_threshold": decision_threshold,
                "calibration_classification": calibration_metrics,
            },
            "visibility": result,
            "persistence": baseline,
            "per_test_trace": per_trace,
        }

    summary: dict[str, object] = {
        "status": "PASS",
        "mechanism": (
            "direct standardized ridge regression from 6DoF history only to "
            "future per-cell contributing_gaussian_fraction"
        ),
        "sequence": sequence,
        "config": {
            "fps": fps,
            "history_ms": history_ms,
            "history_steps": history_steps,
            "horizons_ms": list(horizon_steps),
            "visibility_threshold": visibility_threshold,
            "training_target_mode": target_mode,
            "feature_mode": feature_mode,
            "decision_threshold_search": {
                "minimum": decision_threshold_min,
                "maximum": decision_threshold_max,
                "steps": decision_threshold_steps,
                "selection_data": "two held-out traces within the eight training traces",
                "final_model_data": "all eight training traces",
            },
            "ridge_alpha": ridge_alpha,
            "test_fraction": test_fraction,
            "seed": seed,
        },
        "split": {
            "training_traces": sorted(train_names),
            "test_traces": sorted(test_names),
        },
        "cell_count": len(cell_ids),
        "traces": [
            {
                "name": trace.name,
                "path": str(trace.path.resolve()),
                "split": "train" if trace.name in train_names else "test",
                "duration_s": trace.duration_s,
                "source_rows": trace.source_rows,
                "source_effective_fps": trace.effective_fps,
                "visibility_frames": len(visibility_by_name[trace.name].times_s),
            }
            for trace in traces
        ],
        "horizons": horizon_results,
    }
    write_json_atomic(output / "linear_visibility_summary.json", summary)
    write_csv_atomic(
        output / "per_trace_metrics.csv",
        [
            "horizon_ms",
            "trace",
            "samples",
            "decision_threshold",
            "mse",
            "persistence_mse",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f2",
            "balanced_accuracy",
            "target_visible_cells_per_frame",
            "predicted_cells_per_frame",
            "missed_visible_cells_per_frame",
            "extra_cells_per_frame",
            "persistence_accuracy",
            "persistence_f1",
            "persistence_f2",
            "persistence_missed_cells_per_frame",
            "persistence_extra_cells_per_frame",
        ],
        metrics_rows,
    )
    return summary
