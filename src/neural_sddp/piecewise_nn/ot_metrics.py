# coding=utf-8
# Copyright 2021 The Neural Sddp Authors.
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

"""OT metrics."""

import jax
import jax.numpy as jnp


@jax.jit
def pdist2(points_x, points_y):
  x_norm = jnp.expand_dims(jnp.sum(points_x ** 2, axis=1), 1)
  y_norm = jnp.expand_dims(jnp.sum(points_y ** 2, axis=1), 0)
  cross = jnp.matmul(points_x, points_y.T)
  dist = x_norm - 2 * cross + y_norm
  return dist


@jax.jit
def emd_approx_match(points_x, points_y):
  """Compute approximate match for EMD.

  Args:
    points_x: nxd matrix
    points_y: mxd matrix
  Returns:
    match: n x m matrix for approximated matching
  """
  n = points_x.shape[0]
  m = points_y.shape[0]
  factorl = max(n, m) / n
  factorr = max(n, m) / m
  pairwise_dist2 = pdist2(points_x, points_y)
  saturatedl = jnp.ones(n, dtype=points_x.dtype) * factorl
  saturatedr = jnp.ones(m, dtype=points_y.dtype) * factorr
  match = jnp.zeros((n, m), dtype=points_x.dtype)
  scalars = [-4.0 ** j for j in range(8, -3, -1)]
  scalars[-1] = 0.0
  for level in scalars:
    e_sr = jnp.expand_dims(saturatedr, axis=0)
    # log_sr = jnp.where(e_sr > 0.0, jnp.log(e_sr), -1e10)
    log_sr = jnp.log(e_sr + 1e-30)
    log_weight = pairwise_dist2 * level + log_sr
    weight = jax.nn.softmax(log_weight, axis=-1)
    weight = weight * jnp.expand_dims(saturatedl, axis=1)

    ss = jnp.sum(weight, axis=0) + 1e-9
    ss = jnp.minimum(saturatedr / ss, 1.0)
    weight = weight * jnp.expand_dims(ss, axis=0)
    s = jnp.sum(weight, axis=1)
    ss2 = jnp.sum(weight, axis=0)
    saturatedl = jnp.maximum(saturatedl - s, 0.0)
    match = match + weight
    saturatedr = jnp.maximum(saturatedr - ss2, 0.0)
  match = jax.lax.stop_gradient(match)
  return match


@jax.jit
def emd_approx(points_x, points_y):
  pairwise_dist2 = pdist2(points_x, points_y)
  match = emd_approx_match(points_x, points_y)
  match_cost = jnp.sum(pairwise_dist2 * match)
  return match_cost


@jax.jit
def chamfer_dist(points_x, points_y):
  pairwise_dist2 = pdist2(points_x, points_y)
  match_cost = jnp.sum(jnp.min(pairwise_dist2, axis=-1))
  return match_cost
