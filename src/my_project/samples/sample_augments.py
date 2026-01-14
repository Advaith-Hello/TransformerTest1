"""
Sample augments for usage
Augments can be run with my_project.augment.augment()
All augments are @jit functions from my_project.augment_fns

pad_to: (height, width)
    Pads image equally on both axis with zeros
    Final image will be of shape (height, width)
    This augmentation is fully deterministic

crop_to: (height, width)
    Crops a random square of shape (height, width)
    Final image will be of shape (height, width)

vertical_flip: p
    Has a chance p of vertically flipping the image
    Final image will not change in shape

horizontal_flip: p
    Has a chance p of horizontally flipping the image
    Final image will not change in shape

cutout: proportion
    Zeros out a random square
    The square has shape = floor(proportion * input.shape)
    Final image will not change in shape

images_to_patches: patch_size
    Resizes the image for vision transformers
    Input shape:  (batch_size, height, width, channel)
    Output shape: (batch_size, patch_y, patch_x, pixel_y, pixel_x)
    This augmentation is fully deterministic
    This augmentation does not change the values
"""

from my_project import augment_fns


augments1 = (
    (augment_fns.pad_to, (32, 32)),
    (augment_fns.crop_to, (28, 28)),
    (augment_fns.horizontal_flip, 0.5),
    (augment_fns.cutout, 0.2),
    (augment_fns.images_to_patches, 4),
)

augments2 = (
    (augment_fns.pad_to, (32, 32)),
    (augment_fns.crop_to, (28, 28)),
    (augment_fns.horizontal_flip, 0.5),
    (augment_fns.cutout, 0.2),
)

augments_vit_testing = (
    (augment_fns.images_to_patches, 4),
)
