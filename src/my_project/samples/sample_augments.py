"""
Sample augments for usage
"""

from my_project import augment_fns


augments = (
    (augment_fns.pad_to, (224, 224)),
    (augment_fns.crop_to, (224, 224)),
    (augment_fns.horizontal_flip, 0.5),
    (augment_fns.cutout, 0.2),
)
