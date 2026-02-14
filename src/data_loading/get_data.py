import jax.numpy as jnp

import tensorflow_datasets as tfds

from typing import Any
from jax import Array
from data_loading.data_utils import batch


def get_data(
        batch_size: int,
        dataset: str,
        max_per_split: list[int],
    ) -> tuple[tuple[Any, Array], tuple[Any, Array]]:

    train_ds, test_ds = tfds.load(
        name=dataset,
        split=["train", "test"],
        as_supervised=True,
    )

    train_ds = train_ds.batch(max_per_split[0])
    test_ds  = test_ds.batch(max_per_split[1])

    raw_train_x, raw_train_y = next(iter(tfds.as_numpy(train_ds)))
    raw_test_x, raw_test_y   = next(iter(tfds.as_numpy(test_ds)))

    train_x = jnp.array(raw_train_x, dtype=jnp.float32) / 255.0
    train_y = jnp.array(raw_train_y, dtype=jnp.int32)
    test_x  = jnp.array(raw_test_x, dtype=jnp.float32) / 255.0
    test_y  = jnp.array(raw_test_y, dtype=jnp.int32)

    train_x, train_y = batch(train_x, train_y, batch_size=batch_size)
    test_x, test_y   = batch(test_x, test_y, batch_size=batch_size)

    return (train_x, train_y), (test_x, test_y)
