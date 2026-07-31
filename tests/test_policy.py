import math
import unittest

from fovsim.policy import PolicyConfig, classify_fraction


class PolicyTests(unittest.TestCase):
    def test_threshold_is_inclusive(self) -> None:
        policy = PolicyConfig(threshold=0.5)
        self.assertEqual(classify_fraction(0.499999, policy), (False, 0))
        self.assertEqual(classify_fraction(0.5, policy), (True, 3))
        self.assertEqual(classify_fraction(1.0, policy), (True, 3))

    def test_invalid_fraction_is_rejected(self) -> None:
        for value in (-0.01, 1.01, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                classify_fraction(value, PolicyConfig())


if __name__ == "__main__":
    unittest.main()
