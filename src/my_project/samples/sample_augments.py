"""
Sample augments for usage
"""


augment1 = (
    ("pad_to", (32, 32)),
    ("crop_to", (28, 28)),
    ("horizontal_flip", 0.5),
    ("cutout", (0.15, 0.15)),
    ("gaussian_noise", (0.01, 0.05)),
)
