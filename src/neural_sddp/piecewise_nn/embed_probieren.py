import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Any

class MeinModell(nn.Module):

    @nn.compact
    def __call__(self, x) -> Any:
        emb = nn.Embed(num_embeddings=2, features=128)(x)
        return emb

model = MeinModell()
key = jax.random.PRNGKey(0)
dummy_input = jnp.array([1])
params = model.init(key, dummy_input)
input_a = jnp.array([0])
output = model.apply(params, input_a)
print('Der Output ist')
print(output)
