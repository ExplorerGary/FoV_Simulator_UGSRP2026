"""CellSight-style independent linear prediction of future 6DoF poses."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .io import write_csv_atomic, write_json_atomic
from .prediction import SampledTrace, _split_names, discover_traces

if TYPE_CHECKING:
    import numpy as np


DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
CIRCULAR_NAMES = (
    "x", "y", "z", "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
    "sin_yaw", "cos_yaw",
)


def _circular_pose(dof: "np.ndarray") -> "np.ndarray":
    import numpy as np

    radians = np.deg2rad(dof[..., 3:])
    rotation = np.stack((np.sin(radians), np.cos(radians)), axis=-1)
    rotation = rotation.reshape(*dof.shape[:-1], 6)
    return np.concatenate((dof[..., :3], rotation), axis=-1)


def _pose_windows(
    trace: SampledTrace, history_steps: int, horizon_steps: int
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    encoded = _circular_pose(trace.dof)
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_dof: list[np.ndarray] = []
    for current in range(history_steps, len(encoded) - horizon_steps):
        histories.append(encoded[current - history_steps : current + 1])
        target = current + horizon_steps
        targets.append(encoded[target])
        target_dof.append(trace.dof[target])
    if not histories:
        raise ValueError(f"Trace {trace.name} is too short for the requested windows")
    return np.stack(histories), np.stack(targets), np.stack(target_dof)


def _independent_features(histories: "np.ndarray") -> list["np.ndarray"]:
    """Return one history matrix for each independent circular coordinate."""
    return [histories[:, :, index] for index in range(histories.shape[2])]


def _fit_independent_ols(
    histories: "np.ndarray", targets: "np.ndarray"
) -> list[object]:
    from sklearn.linear_model import LinearRegression

    models: list[object] = []
    for values, target in zip(_independent_features(histories), targets.T):
        models.append(LinearRegression(fit_intercept=True).fit(values, target))
    return models


def _predict_independent_ols(
    models: Sequence[object], histories: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    prediction = np.column_stack(
        [model.predict(values) for model, values in zip(models, _independent_features(histories))]
    )
    for sine_index, cosine_index in ((3, 4), (5, 6), (7, 8)):
        norm = np.hypot(prediction[:, sine_index], prediction[:, cosine_index])
        prediction[:, sine_index] /= np.maximum(norm, 1e-12)
        prediction[:, cosine_index] /= np.maximum(norm, 1e-12)
    return prediction


def _predict_temporal_ols(
    histories: "np.ndarray", horizon_steps: int
) -> "np.ndarray":
    """Match CellSight's per-window LR baseline over the time index."""
    import numpy as np

    if histories.ndim != 3 or histories.shape[1] < 2:
        raise ValueError("Temporal LR requires at least two history samples")
    window = histories.shape[1]
    time = np.arange(window, dtype=np.float64)
    centered_time = time - np.mean(time)
    denominator = float(centered_time @ centered_time)
    centered_values = histories - np.mean(histories, axis=1, keepdims=True)
    slope = np.sum(centered_values * centered_time[None, :, None], axis=1)
    slope /= denominator
    intercept = np.mean(histories, axis=1) - slope * np.mean(time)
    prediction = intercept + slope * (window + horizon_steps - 1)
    for sine_index, cosine_index in ((3, 4), (5, 6), (7, 8)):
        norm = np.hypot(prediction[:, sine_index], prediction[:, cosine_index])
        prediction[:, sine_index] /= np.maximum(norm, 1e-12)
        prediction[:, cosine_index] /= np.maximum(norm, 1e-12)
    return prediction


def _decode_pose(encoded: "np.ndarray") -> "np.ndarray":
    import numpy as np

    angles = np.column_stack(
        [
            np.rad2deg(np.arctan2(encoded[:, sine], encoded[:, cosine]))
            for sine, cosine in ((3, 4), (5, 6), (7, 8))
        ]
    )
    return np.column_stack((encoded[:, :3], angles))


def _angular_error_degrees(prediction: "np.ndarray", target: "np.ndarray") -> "np.ndarray":
    return (prediction - target + 180.0) % 360.0 - 180.0


def _r2(target: "np.ndarray", prediction: "np.ndarray") -> float:
    import numpy as np

    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - residual / total if total > 1e-12 else 0.0


def _metrics(
    prediction: "np.ndarray",
    target: "np.ndarray",
    prediction_encoded: "np.ndarray",
    target_encoded: "np.ndarray",
) -> dict[str, object]:
    import numpy as np

    position_error = prediction[:, :3] - target[:, :3]
    angle_error = _angular_error_degrees(prediction[:, 3:], target[:, 3:])
    per_dof: dict[str, dict[str, float]] = {}
    for index, name in enumerate(DOF_NAMES[:3]):
        error = position_error[:, index]
        per_dof[name] = {
            "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": _r2(target[:, index], prediction[:, index]),
            "unit": "cm",
        }
    for offset, name in enumerate(DOF_NAMES[3:]):
        error = angle_error[:, offset]
        aligned_prediction = target[:, 3 + offset] + error
        per_dof[name] = {
            "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": _r2(target[:, 3 + offset], aligned_prediction),
            "unit": "degree",
        }
    circular_r2 = {
        name: _r2(target_encoded[:, index], prediction_encoded[:, index])
        for index, name in enumerate(CIRCULAR_NAMES[3:], start=3)
    }
    return {
        "samples": int(len(target)),
        "position_mse_cm2": float(np.mean(position_error**2)),
        "position_rmse_cm": float(np.sqrt(np.mean(position_error**2))),
        "position_mae_cm": float(np.mean(np.abs(position_error))),
        "position_r2": _r2(target[:, :3].reshape(-1), prediction[:, :3].reshape(-1)),
        "orientation_mse_deg2": float(np.mean(angle_error**2)),
        "orientation_rmse_deg": float(np.sqrt(np.mean(angle_error**2))),
        "orientation_mae_deg": float(np.mean(np.abs(angle_error))),
        "orientation_circular_r2": _r2(
            target_encoded[:, 3:].reshape(-1), prediction_encoded[:, 3:].reshape(-1)
        ),
        "per_dof": per_dof,
        "per_circular_coordinate_r2": circular_r2,
    }


def _metric_row(step: str, trace: str, metrics: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "trace": trace,
        "samples": metrics["samples"],
        "position_mse_cm2": metrics["position_mse_cm2"],
        "position_rmse_cm": metrics["position_rmse_cm"],
        "position_mae_cm": metrics["position_mae_cm"],
        "position_r2": metrics["position_r2"],
        "orientation_mse_deg2": metrics["orientation_mse_deg2"],
        "orientation_rmse_deg": metrics["orientation_rmse_deg"],
        "orientation_mae_deg": metrics["orientation_mae_deg"],
        "orientation_circular_r2": metrics["orientation_circular_r2"],
    }
    per_dof = metrics["per_dof"]
    assert isinstance(per_dof, dict)
    for name in DOF_NAMES:
        values = per_dof[name]
        row[f"{name}_mse"] = values["mse"]
        row[f"{name}_r2"] = values["r2"]
    return row


def run_dof_lr_experiment(
    *,
    trace_dir: str | Path,
    output_dir: str | Path,
    sequence: str = "BiancaGolden_CircleTurns",
    fps: float = 30.0,
    history_ms: int = 500,
    horizon_ms: int = 100,
    test_fraction: float = 0.2,
    seed: int = 20260731,
    expected_traces: int | None = 10,
) -> dict[str, object]:
    """Train and evaluate a CellSight-style independent-coordinate OLS model."""
    import numpy as np

    if fps <= 0 or history_ms <= 0 or horizon_ms <= 0:
        raise ValueError("fps, history_ms, and horizon_ms must be positive")
    history_steps = int(round(history_ms * fps / 1000.0))
    horizon_steps = int(round(horizon_ms * fps / 1000.0))
    traces = discover_traces(trace_dir, sequence=sequence, fps=fps)
    if expected_traces is not None and len(traces) != expected_traces:
        raise ValueError(f"Expected {expected_traces} traces, found {len(traces)}")
    train_names, test_names = _split_names(
        [trace.name for trace in traces], test_fraction, seed
    )
    windows = {
        trace.name: _pose_windows(trace, history_steps, horizon_steps)
        for trace in traces
    }
    train_histories = np.concatenate([windows[name][0] for name in sorted(train_names)])
    train_targets = np.concatenate([windows[name][1] for name in sorted(train_names)])
    models = _fit_independent_ols(train_histories, train_targets)

    test_histories = np.concatenate([windows[name][0] for name in sorted(test_names)])
    test_targets_encoded = np.concatenate([windows[name][1] for name in sorted(test_names)])
    test_targets = np.concatenate([windows[name][2] for name in sorted(test_names)])
    lr_encoded = _predict_independent_ols(models, test_histories)
    lr_prediction = _decode_pose(lr_encoded)
    persistence_encoded = test_histories[:, -1, :]
    persistence_prediction = _decode_pose(persistence_encoded)

    lr_metrics = _metrics(lr_prediction, test_targets, lr_encoded, test_targets_encoded)
    persistence_metrics = _metrics(
        persistence_prediction, test_targets, persistence_encoded, test_targets_encoded
    )
    per_trace: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for name in sorted(test_names):
        histories, encoded_target, dof_target = windows[name]
        trace_lr_encoded = _predict_independent_ols(models, histories)
        trace_persistence_encoded = histories[:, -1, :]
        trace_lr = _metrics(
            _decode_pose(trace_lr_encoded), dof_target, trace_lr_encoded, encoded_target
        )
        trace_persistence = _metrics(
            _decode_pose(trace_persistence_encoded), dof_target,
            trace_persistence_encoded, encoded_target,
        )
        per_trace[name] = {"lr": trace_lr, "persistence": trace_persistence}
        rows.extend(
            (_metric_row("persistence", name, trace_persistence), _metric_row("lr", name, trace_lr))
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_arrays: dict[str, np.ndarray] = {}
    for index, (name, model) in enumerate(zip(CIRCULAR_NAMES, models)):
        model_arrays[f"{name}_coefficient"] = np.asarray(model.coef_)
        model_arrays[f"{name}_intercept"] = np.asarray([model.intercept_])
    np.savez_compressed(output / "dof_lr_model.npz", **model_arrays)
    write_csv_atomic(output / "per_trace_metrics.csv", list(rows[0]), rows)
    step_rows = [
        _metric_row("step_0_persistence", "ALL_TEST_TRACES", persistence_metrics),
        _metric_row("step_1_cellsight_lr", "ALL_TEST_TRACES", lr_metrics),
    ]
    write_csv_atomic(output / "step_metrics.csv", list(step_rows[0]), step_rows)
    summary: dict[str, object] = {
        "status": "PASS",
        "mechanism": "independent ordinary linear regression per circular 6DoF coordinate",
        "config": {
            "fps": fps,
            "history_ms": history_ms,
            "history_steps": history_steps,
            "history_samples_including_current": history_steps + 1,
            "horizon_ms": horizon_ms,
            "horizon_steps": horizon_steps,
            "fit_intercept": True,
            "rotation_encoding": "per_angle_sin_cos",
            "extra_motion_features": False,
            "r2_definition": {
                "position": "ordinary coefficient of determination",
                "orientation_primary": "coefficient of determination in sine/cosine domain",
                "per_angle": "coefficient of determination after shortest-arc prediction alignment",
            },
        },
        "split": {
            "seed": seed,
            "training_traces": sorted(train_names),
            "test_traces": sorted(test_names),
            "training_samples": int(len(train_histories)),
            "test_samples": int(len(test_histories)),
        },
        "steps": {
            "step_0_persistence": persistence_metrics,
            "step_1_cellsight_lr": lr_metrics,
        },
        "delta_lr_minus_persistence": {
            "position_mse_cm2": float(
                lr_metrics["position_mse_cm2"] - persistence_metrics["position_mse_cm2"]
            ),
            "orientation_mse_deg2": float(
                lr_metrics["orientation_mse_deg2"] - persistence_metrics["orientation_mse_deg2"]
            ),
            "position_r2": float(lr_metrics["position_r2"] - persistence_metrics["position_r2"]),
            "orientation_circular_r2": float(
                lr_metrics["orientation_circular_r2"]
                - persistence_metrics["orientation_circular_r2"]
            ),
        },
        "per_test_trace": per_trace,
    }
    write_json_atomic(output / "dof_lr_evaluation.json", summary)
    return summary


def run_cellsight_lr_window_sweep(
    *,
    trace_dir: str | Path,
    output_dir: str | Path,
    history_samples: Sequence[int] = (30, 90),
    sequence: str = "BiancaGolden_CircleTurns",
    fps: float = 30.0,
    horizon_ms: int = 100,
    test_fraction: float = 0.2,
    seed: int = 20260731,
    expected_traces: int | None = 10,
) -> dict[str, object]:
    """Evaluate source-compatible CellSight LR30/LR90 temporal extrapolation."""
    import numpy as np

    if not history_samples or any(value < 2 for value in history_samples):
        raise ValueError("history_samples must contain values of at least two")
    horizon_steps = int(round(horizon_ms * fps / 1000.0))
    traces = discover_traces(trace_dir, sequence=sequence, fps=fps)
    if expected_traces is not None and len(traces) != expected_traces:
        raise ValueError(f"Expected {expected_traces} traces, found {len(traces)}")
    train_names, test_names = _split_names(
        [trace.name for trace in traces], test_fraction, seed
    )
    traces_by_name = {trace.name: trace for trace in traces}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants: dict[str, object] = {}
    aggregate_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    common_history_steps = max(history_samples) - 1
    for samples in history_samples:
        history_steps = samples - 1
        windows = {}
        prefix_to_drop = common_history_steps - history_steps
        for name in sorted(test_names):
            history, encoded_target, dof_target = _pose_windows(
                traces_by_name[name], history_steps, horizon_steps
            )
            windows[name] = (
                history[prefix_to_drop:],
                encoded_target[prefix_to_drop:],
                dof_target[prefix_to_drop:],
            )
        histories = np.concatenate([windows[name][0] for name in sorted(test_names)])
        encoded_target = np.concatenate(
            [windows[name][1] for name in sorted(test_names)]
        )
        dof_target = np.concatenate([windows[name][2] for name in sorted(test_names)])
        persistence_encoded = histories[:, -1, :]
        lr_encoded = _predict_temporal_ols(histories, horizon_steps)
        persistence = _metrics(
            _decode_pose(persistence_encoded), dof_target,
            persistence_encoded, encoded_target,
        )
        lr = _metrics(_decode_pose(lr_encoded), dof_target, lr_encoded, encoded_target)
        label = f"LR{samples}"
        aggregate_rows.extend(
            (
                _metric_row(f"{label}_persistence", "ALL_TEST_TRACES", persistence),
                _metric_row(label, "ALL_TEST_TRACES", lr),
            )
        )
        per_trace: dict[str, object] = {}
        for name in sorted(test_names):
            trace_history, trace_target_encoded, trace_target_dof = windows[name]
            trace_lr_encoded = _predict_temporal_ols(trace_history, horizon_steps)
            trace_lr = _metrics(
                _decode_pose(trace_lr_encoded), trace_target_dof,
                trace_lr_encoded, trace_target_encoded,
            )
            per_trace[name] = trace_lr
            trace_rows.append(_metric_row(label, name, trace_lr))
        variants[label] = {
            "history_samples": samples,
            "history_span_ms": 1000.0 * (samples - 1) / fps,
            "persistence": persistence,
            "lr": lr,
            "per_test_trace": per_trace,
        }
    write_csv_atomic(
        output / "window_metrics.csv", list(aggregate_rows[0]), aggregate_rows
    )
    write_csv_atomic(
        output / "per_trace_window_metrics.csv", list(trace_rows[0]), trace_rows
    )
    summary: dict[str, object] = {
        "status": "PASS",
        "mechanism": (
            "CellSight source-compatible per-window ordinary least-squares "
            "time extrapolation for each circular 6DoF coordinate"
        ),
        "config": {
            "fps": fps,
            "horizon_ms": horizon_ms,
            "horizon_steps": horizon_steps,
            "history_samples": list(history_samples),
            "common_evaluation_start_samples": max(history_samples),
            "rotation_encoding": "per_angle_sin_cos",
            "fit_intercept": True,
            "cross_dof_features": False,
            "trained_cross_trace_model": False,
        },
        "split": {
            "seed": seed,
            "training_traces_reserved_but_not_used_by_local_ols": sorted(train_names),
            "test_traces": sorted(test_names),
        },
        "variants": variants,
    }
    write_json_atomic(output / "cellsight_lr_window_sweep.json", summary)
    return summary
