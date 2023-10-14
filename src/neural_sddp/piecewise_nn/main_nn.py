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

"""Learn from solved sddp instances."""

import enum
from multiprocessing import dummy
from pickletools import optimize
from statistics import mode
from absl import app
from absl import flags
import random
import numpy as np
from typing import Any, Sequence
import functools
import jax
import jax.numpy as jnp
from neural_sddp.piecewise_nn.cond_piecewise_nn import CondPiecewiseNN

import optax

FLAGS = flags.FLAGS
# flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_string('loss_type', 'emd', 'mse/emd')



def main(argv):
  del argv
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  key = jax.random.PRNGKey(1)

  num_vars = 1
  num_pieces = 5
  num_stages = 2
  model = CondPiecewiseNN(num_vars=num_vars,
                          num_stages=2,
                          hidden_size=128,
                          num_pieces=num_pieces,
                          num_layers=1)
  # Output ist (num_pieces, num_vars + 1 (für intercept))
  # num_pieces = anzahl geraden der stückweise linearen Funktion
  # also in meinem Fall meist (num_pieces, 2)
  #dummy_input = jnp.ones([1, 10])
  dummy_stage = np.array([1])
  u = 10
  w = 8
  r = 5
  q = 7
  c = 11
  prob_mean = 3
  prob_sigma = 3
  input_vector = [u, w, r, q, c, prob_mean, prob_sigma]
  dummy_input = jnp.array(input_vector)
  params = model.init(key, dummy_input, dummy_stage)
  optimizer = optax.adam(0.02)
  value_func = jnp.array([(2, 1) for piece in range(num_pieces)])
  label = jnp.reshape(value_func, (-1,  value_func.shape[0], value_func.shape[1]))
  stage = dummy_stage
  params, loss = fit_model(model, dummy_input, stage, label, params, optimizer)
  print('model loss', loss)
  print('apply von modell')
  output = model.apply(params, dummy_input, stage)
  print('dies ist der outpur')
  print(output)
  print('ende von apply von modell')
  # dummy_out, params = model.init_with_output(key, dummy_input,
  #                                            jnp.array([1], dtype=jnp.int32))
  del params


if __name__ == '__main__':
  app.run(main)