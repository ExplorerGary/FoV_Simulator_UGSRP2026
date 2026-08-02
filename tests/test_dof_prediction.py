import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(
    NUMPY_AVAILABLE and SKLEARN_AVAILABLE,
    "DoF prediction requires numpy and scikit-learn",
)
class DofPredictionTests(unittest.TestCase):
    def test_independent_lr_predicts_linear_motion_across_angle_wrap(self) -> None:
        import numpy as np

        from fovsim.dof_prediction import (
            _circular_pose,
            _decode_pose,
            _fit_independent_ols,
            _predict_independent_ols,
            _predict_temporal_ols,
        )

        histories = []
        targets = []
        for start in range(20):
            time = np.arange(16, dtype=float)
            dof = np.column_stack(
                (
                    start + time,
                    2.0 * time,
                    -time,
                    170.0 + time,
                    -20.0 + 0.5 * time,
                    -179.0 + 2.0 * time,
                )
            )
            future = np.asarray(
                [start + 18.0, 36.0, -18.0, 188.0, -11.0, -143.0]
            )
            histories.append(_circular_pose(dof))
            targets.append(_circular_pose(future[None, :])[0])
        history = np.stack(histories)
        target = np.stack(targets)
        models = _fit_independent_ols(history, target)
        prediction = _predict_independent_ols(models, history)
        decoded = _decode_pose(prediction)
        np.testing.assert_allclose(decoded[:, :3], np.asarray(targets)[:, :3], atol=1e-9)
        angle_error = (decoded[:, 3:] - np.asarray([188.0, -11.0, -143.0]) + 180) % 360 - 180
        np.testing.assert_allclose(angle_error, 0.0, atol=0.1)

        temporal = _predict_temporal_ols(history, horizon_steps=3)
        temporal_decoded = _decode_pose(temporal)
        np.testing.assert_allclose(
            temporal_decoded[:, :3], np.asarray(targets)[:, :3], atol=1e-9
        )
        temporal_angle_error = (
            temporal_decoded[:, 3:]
            - np.asarray([188.0, -11.0, -143.0])
            + 180
        ) % 360 - 180
        # Linear extrapolation in sine/cosine space approximates constant
        # angular velocity; atan2 keeps the result on the correct wrap branch.
        np.testing.assert_allclose(temporal_angle_error, 0.0, atol=1.0)
