"""
Sample augments for usage
"""


augment1 = (
    ("pad_to", (32, 32)),
    ("crop_to", (28, 28)),
    ("horizontal_flip", 0.5),
    ("translate", (1, 3)),
    ("rotate", (-12, 12)),
    ("scale", (0.9, 1.1)),
    ("cutout", (0.1, 0.2)),
    ("gaussian_noise", (0.01, 0.05)),
)
