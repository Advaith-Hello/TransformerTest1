"""
Sample augments for usage
"""

from my_project import augment_fns


augments1 = (
    (augment_fns.pad_to, (32, 32)),
    (augment_fns.crop_to, (28, 28)),
    (augment_fns.horizontal_flip, 0.5),
    (augment_fns.cutout, 0.2),
    (augment_fns.images_to_patches, 4),
)

augments_vit_testing = (
    (augment_fns.images_to_patches, 4),
)
