import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

import optax
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from jax import vmap
from tqdm import tqdm

from data_loading.augment import augment
from models import sample_models
from models import sample_augments

from training import loss as loss_fns
from training import parameterize
from training import forward
from data_loading import augment, get_data



BATCH_SIZE = 256

(train_x, train_y), (test_x, test_y) = get_data.get_data(
    batch_size=BATCH_SIZE,
    dataset="mnist",
    max_per_split=[60_000, 10_000]
)

print("Data shapes:")
print("\t", train_x.shape, " ", train_y.shape, sep="")
print("\t", test_x.shape,  " ", test_y.shape,  sep="")
print("")


key = jrd.PRNGKey(0)
model = sample_models.model1
augments = sample_augments.augments_ViT_mnist_1
params = parameterize.parameterize(model, init_key=0)


@jit
def train_step(carry, ds):
    curr_params, curr_opt_state, curr_key = carry
    augment_x, curr_key = augment.augment(ds[0], curr_key, augments)
    loss, grads = loss_fns.cross_entropy_loss_value_and_grad(augment_x, ds[1], curr_params, model)
    updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
    curr_params = optax.apply_updates(curr_params, updates)
    return (curr_params, curr_opt_state, curr_key), loss


eval_loss_fn = jit(vmap(
    lambda x, y, p: loss_fns.cross_entropy_loss(x, y, p, model),
    in_axes=(0, 0, None)
))

eval_forward_fn = jit(vmap(
    lambda x, p: forward.forward(x, p, model),
    in_axes=(0, None)
))

epochs = 100
num_batches = int(train_x.shape[0])

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=((epochs // 4) * num_batches),
    decay_steps=(epochs * num_batches),
    end_value=1e-5
)

optimizer = optax.adamw(
    learning_rate=schedule,
    weight_decay=0.05,
)

opt_state = optimizer.init(params)

total_train_losses = np.empty(epochs)
total_test_losses = np.empty(epochs)
total_train_accuracy = np.empty(epochs)
total_test_accuracy = np.empty(epochs)

for i in tqdm(range(epochs)):
    (params, opt_state, key), curr_loss = lax.scan(
        train_step,
        (params, opt_state, key),
        (train_x, train_y),
    )

    aug_train_x, _ = augment.augment_ds(
        train_x, key,
        sample_augments.augments_vit_testing
    )
    aug_test_x, _ = augment.augment_ds(
        test_x, key,
        sample_augments.augments_vit_testing
    )

    train_logits = eval_forward_fn(aug_train_x, params)
    train_pred = jnp.argmax(train_logits, axis=-1)

    test_logits = eval_forward_fn(aug_test_x, params)
    test_pred = jnp.argmax(test_logits, axis=-1)

    total_train_losses[i] = jnp.mean(curr_loss)
    total_test_losses[i] = jnp.mean(eval_loss_fn(aug_test_x, test_y, params))
    total_test_accuracy[i] = jnp.mean(test_pred == test_y)
    total_train_accuracy[i] = jnp.mean(train_pred == train_y)


fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

axs[0].set_title(f"Train loss (blue) vs Test loss (red) over {epochs} epochs")
axs[0].plot(total_train_losses, color='blue')
axs[0].plot(total_test_losses, color='red')

axs[1].set_title(f"Train accuracy (blue) vs Test accuracy (red) over {epochs} epochs")
axs[1].plot(total_train_accuracy, color='blue')
axs[1].plot(total_test_accuracy, color='red')

plt.tight_layout()
plt.show()
