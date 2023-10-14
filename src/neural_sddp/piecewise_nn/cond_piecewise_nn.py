# coding=utf-8 # Copyright 2021 The Neural Sddp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conditional piecewise linear nn."""

from typing import Any, Callable, Sequence
import functools

from absl import flags
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from neural_sddp.piecewise_nn.ot_metrics import emd_approx


class MLP(nn.Module):
  sizes: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for size in self.sizes[:-1]:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    return nn.Dense(self.sizes[-1])(x)


class CondPiecewiseNN(nn.Module):
  """Generate piecewise linear functions based on conditional features."""
  num_vars: int
  num_stages: int
  hidden_size: int
  num_pieces: int
  num_layers: int
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, cond_feat, stage_idx):
    # Wenn Embedding 2 oder kleiner, ist eh klar, dass Trainingsinstanzen zu der einen
    # verfügbaren Stufe gehören
    cond_feat = nn.Dense(self.hidden_size)(cond_feat)
    if self.num_stages <= 2:
        time_embedding = jnp.array([0])
    else:
        time_embedding = nn.Embed(num_embeddings=self.num_stages - 1,
                              features=self.hidden_size)(stage_idx)
        cond_feat = cond_feat + time_embedding
    # Festlegen von den Output Sizes der jeweiligen Layer
    l = [self.hidden_size] * self.num_layers
    l.append((self.num_vars + 1) * self.num_pieces)
    joint_param = MLP(tuple(l))(cond_feat)
    params = jnp.reshape(joint_param, (-1, self.num_pieces, self.num_vars + 1))
    return params

  @staticmethod
  def emd_approx(pred_params, target_pieces):
    """Calculate batched emd distance.

    Args:
      pred_params: pred_params of shape [bsize, self.n_pieces, n_vars + 1]
      target_pieces: tensor of shape [bsize, n_pieces, n_vars + 1]
    Returns:
      loss
    """
    # print("target_pieces")
    # print(target_pieces.shape)
    # print("pred data")
    # print(pred_params.shape)
    return jax.vmap(emd_approx)(target_pieces, pred_params)

  def mse(self, pred_params, target_pieces):
    """Calculate mean square error."""
    n = min(pred_params.shape[1], target_pieces.shape[1])
    dist = pred_params[:, -n:, :] - target_pieces[:, -n:, :]
    dist = jnp.sum(dist ** 2, axis=[1, 2])
    return dist


@functools.partial(jax.jit, static_argnums=0)
def jit_apply(model, params, feat, stage):
  return model.apply(params, feat, stage)

# Model ist das Neuronale Netz
# target sind die richtigen Wertfunktionen also Vektor mit (beta, alpha)
@functools.partial(jax.jit, static_argnums=0)
def eval_loss(model, params, feat, stage, target):
  # verwendet das Modell mit params und feat und stage als eingabe

  pred_params = jit_apply(model, params, feat, stage)
  # print("eval loss pred params shape")
  # print(pred_params.shape)
  # print(target.shape)
  dist = model.emd_approx(pred_params, target)
  return jnp.mean(dist)


# @functools.partial(jax.jit, static_argnums=0)
# def train_step(model, feat, stage, target, optimizer):

#   def loss(params):
#     return eval_loss(model, params, feat, stage, target)
#   # optimizer.target ist eig ein dict bzw das was von flax.linen.Module.init zurück kommt, also params
#   l, grads = jax.value_and_grad(loss)(optimizer.target)
#   optimizer = optimizer.apply_gradient(grads)
#   return l, optimizer


@functools.partial(jax.jit, static_argnums=[0, 2, 4])
def fit_model(
  model: CondPiecewiseNN,
  params: optax.Params,
  optimizer: optax.GradientTransformation,
  n_epochs: int,
  tol: float,
  feature: Any,
  stage: Any,
  target: Any,
) -> optax.Params:

  opt_state = optimizer.init(params)

  
  params, loss = train_loop(model, params, n_epochs, tol, feature, stage, target, opt_state, train_step)
  return params, loss

@functools.partial(jax.jit, static_argnums=[0, 5])
def train_step(model, feat, stage, target, params, optimizer, opt_state):
  def loss(params):
    return eval_loss(model, params, feat, stage, target)
  l, grads = jax.value_and_grad(loss)(params)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state, l

def train_loop(
  model,
  params,
  optimizer,
  n_epochs,
  tol,
  feature,
  stage,
  target,
):
    opt_state = optimizer.init(params)
    for epoch in range(n_epochs):
        params, opt_state, loss = train_step(
          model,
          feature,
          stage,
          target,
          params,
          optimizer,
          opt_state
        )
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, loss: {loss}')
        if loss < tol:
            print(f'The Loss {loss} lower than tolerance: {tol}')
            break
    return params,loss