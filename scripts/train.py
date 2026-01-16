import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

import optax
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from jax import vmap
from tqdm import tqdm

from my_project.augment import augment
from my_project.samples import sample_models
from my_project.samples import sample_augments

from my_project import loss as loss_fns
from my_project import parameterize
from my_project import get_data
from my_project import forward
from my_project import augment


(train_x, train_y), (test_x, test_y) = get_data.get_data(
    batch_size=256,
    dataset="fashion_mnist",
    max_per_split=[60_000, 10_000]
)

print("Data shapes:")
print("\t", train_x.shape, " ", train_y.shape, sep="")
print("\t", test_x.shape,  " ", test_y.shape,  sep="")
print("")


key = jrd.PRNGKey(0)
model = sample_models.model1
augments = sample_augments.augments_ViT_fashion_mnist_1
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
    in_axes=(0, 0)
))

eval_forward_fn = jit(vmap(
    lambda x, p: forward.forward(x, p, model),
    in_axes=0
))

epochs = 100

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
    (params, opt_state, key), total_train_losses[i] = lax.scan(
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

    total_test_losses[i] = np.array(eval_loss_fn(aug_test_x, test_y, params))

    train_logits = eval_forward_fn(aug_train_x, params)
    train_pred = jnp.argmax(train_logits, axis=-1)
    total_train_accuracy[i] = jnp.mean(train_pred == train_y)

    test_logits = eval_forward_fn(aug_test_x, params)
    test_pred = jnp.argmax(test_logits, axis=-1)
    total_test_accuracy[i] = jnp.mean(test_pred == test_y)


fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

axs[0].set_title(f"Train loss (blue) vs Test loss (red) over {epochs} epochs")
axs[0].plot(np.mean(total_train_losses, axis=1), color='blue')
axs[0].plot(np.mean(total_test_losses, axis=1), color='red')

axs[1].set_title(f"Train accuracy (blue) vs Test accuracy (red) over {epochs} epochs")
axs[1].plot(np.mean(total_train_accuracy, axis=1), color='blue')
axs[1].plot(np.mean(total_test_accuracy, axis=1), color='red')

plt.tight_layout()
plt.show()
