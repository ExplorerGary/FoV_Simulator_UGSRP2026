import unittest

from fovsim.qoe import alex_lpips_crop_is_supported


class QoeTests(unittest.TestCase):
    def test_alex_lpips_rejects_undersized_foreground_crop(self) -> None:
        self.assertFalse(alex_lpips_crop_is_supported(2, 15))
        self.assertFalse(alex_lpips_crop_is_supported(7, 2))
        self.assertFalse(alex_lpips_crop_is_supported(2, 9))

    def test_alex_lpips_accepts_minimum_and_larger_crop(self) -> None:
        self.assertTrue(alex_lpips_crop_is_supported(32, 32))
        self.assertTrue(alex_lpips_crop_is_supported(1080, 1920))


if __name__ == "__main__":
    unittest.main()
