import jax.lax as lax
import jax.numpy as jnp

import optax
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from jax import jit
from tqdm import tqdm

from my_project import loss as loss_fns
from my_project import parameterize
from my_project import forward


tf.config.set_visible_devices([], "GPU")


batch_size = 256
patch_size_dim = 4
num_patches_per_dim = 28 // patch_size_dim  # 7
num_patches = num_patches_per_dim ** 2  # 49
patch_size = patch_size_dim * patch_size_dim  # 16

train_ds, test_ds = tfds.load(
    "fashion_mnist",
    split=["train", "test"],
    as_supervised=True,
)

def augment(img, lbl):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_with_crop_or_pad(img, 32, 32)
    img = tf.image.random_crop(img, size=[28, 28, 1])
    return img, lbl

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.batch(60000)
test_ds  = test_ds.batch(10000)

raw_train_x, raw_train_y = next(iter(tfds.as_numpy(train_ds)))
raw_test_x, raw_test_y   = next(iter(tfds.as_numpy(test_ds)))

train_x = jnp.array(raw_train_x, dtype=jnp.float32) / 255.0
train_y = jnp.array(raw_train_y, dtype=jnp.int32)

test_x = jnp.array(raw_test_x, dtype=jnp.float32) / 255.0
test_y = jnp.array(raw_test_y, dtype=jnp.int32)


def images_to_patches(x):
    x = x.reshape(
        x.shape[0],
        num_patches_per_dim,
        patch_size_dim,
        num_patches_per_dim,
        patch_size_dim
    )
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(x.shape[0], num_patches, patch_size)
    return x


num_train_batches = train_x.shape[0] // batch_size
num_test_batches = test_x.shape[0] // batch_size

train_x = train_x[:num_train_batches * batch_size]
train_y = train_y[:num_train_batches * batch_size]
test_x = test_x[:num_test_batches * batch_size]
test_y = test_y[:num_test_batches * batch_size]

train_x = images_to_patches(train_x).reshape(num_train_batches, batch_size, num_patches, patch_size)
train_y = train_y.reshape(num_train_batches, batch_size)

test_x = images_to_patches(test_x).reshape(num_test_batches, batch_size, num_patches, patch_size)
test_y = test_y.reshape(num_test_batches, batch_size)

# (num_batches, batch_size, num_patches, patch_size)
print("Data shapes:")
print("\t", train_x.shape, " ", train_y.shape, sep="")
print("\t", test_x.shape, " ", test_y.shape, sep="")


structure = (
    ("linear", (16, 64)),
    ("positional_embedding", (49, 64)),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("linear", (64, 128)),
    ("relu", ()),
    ("linear", (128, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("attention", (64, 64, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("collect_residual", ()),
    ("linear", (64, 128)),
    ("relu", ()),
    ("linear", (128, 64)),
    ("add_residual", ()),
    ("layer_norm", (64,)),

    ("mean_pool", ()),
    ("linear", (64, 10)),
)

params = parameterize.parameterize(structure, init_key=0)


@jit
def train_step(carry, ds):
    curr_params, curr_opt_state = carry
    loss, grads = loss_fns.cross_entropy_loss_value_and_grad(*ds, curr_params, structure)
    updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
    curr_params = optax.apply_updates(curr_params, updates)
    return (curr_params, curr_opt_state), loss


@jit
def calc_accuracy(x, y, curr_params):
    logits = forward.forward(x, curr_params, structure)
    x_hat = jnp.argmax(logits, axis=1)
    return jnp.sum(y == x_hat) / x_hat.shape[0]


epochs = 50

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=10 * len(train_x),
    decay_steps=len(train_x) * epochs,
    end_value=1e-5
)

optimizer = optax.adamw(
    learning_rate=schedule,
    weight_decay=0.05,
)

opt_state = optimizer.init(params)

total_train_losses = np.empty(shape=(epochs, train_x.shape[0]))
total_test_losses = np.empty(shape=(epochs, test_x.shape[0]))
total_train_accuracy = np.empty(shape=(epochs, train_x.shape[0]))
total_test_accuracy = np.empty(shape=(epochs, test_x.shape[0]))

for i in tqdm(range(epochs)):
    (params, opt_state), total_train_losses[i] = lax.scan(
        train_step,
        (params, opt_state),
        (train_x, train_y),
    )

    total_test_losses[i] = np.array([
        loss_fns.cross_entropy_loss(test_x[j], test_y[j], params, structure)
        for j in range(test_x.shape[0])
    ])

    total_train_accuracy[i] = np.array([
        calc_accuracy(train_x[j], train_y[j], params)
        for j in range(train_x.shape[0])
    ])

    total_test_accuracy[i] = np.array([
        calc_accuracy(test_x[j], test_y[j], params)
        for j in range(test_x.shape[0])
    ])


fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

axs[0].set_title(f"Train loss (blue) vs Test loss (red) over {epochs} epochs")
axs[0].plot(np.mean(total_train_losses, axis=1), color='blue')
axs[0].plot(np.mean(total_test_losses, axis=1), color='red')

axs[1].set_title(f"Train accuracy (blue) vs Test accuracy (red) over {epochs} epochs")
axs[1].plot(np.mean(total_train_accuracy, axis=1), color='blue')
axs[1].plot(np.mean(total_test_accuracy, axis=1), color='red')

plt.show()
