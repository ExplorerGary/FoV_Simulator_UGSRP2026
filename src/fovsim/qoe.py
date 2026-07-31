"""Pure helpers for QoE metric validity decisions."""

LPIPS_ALEX_MIN_CROP_SIZE = 32


def alex_lpips_crop_is_supported(height: int, width: int) -> bool:
    """Return whether an unscaled crop can pass through AlexNet LPIPS.

    AlexNet's strided convolution and pooling stack collapses spatial axes
    below roughly 31 pixels. We use 32 as a stable boundary across versions.
    """

    return (
        int(height) >= LPIPS_ALEX_MIN_CROP_SIZE
        and int(width) >= LPIPS_ALEX_MIN_CROP_SIZE
    )
