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

"""Synthetic Data Generator for Time Series Data."""

import abc
from typing import Sequence

import attr
import immutabledict
import numpy as np


EPSILON = 0.001


@attr.s(auto_attribs=True)
class GeneratorParams:
  low: float = 0.0
  high: float = 0.0
  mean: float = 0.0
  std_dev: float = 0.0
  ar_order: int = 2
  coefficients: np.ndarray = None
  intercepts: np.ndarray = None
  init_point: np.ndarray = None
  std_err: np.ndarray = None


class TimeSeriesGenerator(metaclass=abc.ABCMeta):
  """Base time series generator class."""

  def __init__(self, init_params, seed):
    """Initialize generator.

    Args:
      init_params: GeneratorParams, initialization parameters.
      seed: random seed to control reproducibility.
    """
    self._params = init_params
    self._seed = seed
    np.random.seed(seed=seed)

  @abc.abstractmethod
  def get_next_point(self):
    """Get the next point in the series.

    Returns: A scalar value representing the next point in the series.
    """
    pass

  @abc.abstractmethod
  def get_series(self,
                 length,
                 shape = (),
                 cached = False):
    """Get a series of fixed length.

    Args:
      length: length of this series.
      shape: shape of np.array per stage.
      cached: use a cached value.

    Returns:
      If `shape` is not specified, returns np.array with shape (length,).
      If `shape` is specified, returns np.array with shape (length, shape).
      For example, if length=3, shape=(5,6), the returned np.array shape is
      (3, 5, 6).
    """
    pass

  def get_params(self):
    """Returns the initialization parameters of the generator."""
    return self._params


class DeltaTimeSeriesGenerator(TimeSeriesGenerator):
  """Sampled values are EPSILON more than the original time series."""

  def __init__(self, original_gen):
    """Initialize generator."""
    self._original_gen = original_gen

  def get_series(self,
                 length,
                 shape = (),
                 cached = False):
    return self._original_gen.get_series(length, shape, cached) + EPSILON

  def get_next_point(self):
    raise ValueError('not supported')


class NegationTimeSeriesGenerator(TimeSeriesGenerator):
  """Sampled values are negation of original time series."""

  def __init__(self, original_gen):
    """Initialize generator."""
    self._original_gen = original_gen

  def get_series(self,
                 length,
                 shape = (),
                 cached = False):
    return -self._original_gen.get_series(length, shape, cached)

  def get_next_point(self):
    raise ValueError('not supported')


class AutoRegressiveGenerator(TimeSeriesGenerator):
  """Data is sampled from an auto regressive process."""

  def __init__(self, init_params, seed):
    """Initialize AutoRegressiveGenerator.

    Args:
      init_params: GeneratorParams, initialization parameters.
      seed: to control the randomness of the data generator.
    """
    super().__init__(init_params=init_params, seed=seed)
    # Whether the result is cached.
    self._cache = None

  def get_next_point(self):
    """Gets the next point in the series."""
    raise ValueError('AutoRegressiveGenerator does not support get_next_point.')

  def get_series(self,
                 length,
                 shape = (),
                 cached = False):
    if cached:
      if self._cache is None:
        raise ValueError('_cache can not be none.')
      return self._cache
    ar_order = self._params.init_point.shape[1]
    series_list = []
    historical_price = [self._params.init_point[:, r] for r in range(ar_order)]

    for _ in range(length):
      current_mean = sum([
          np.multiply(historical_price[r], self._params.coefficients[:, r])
          for r in range(ar_order)
      ])
      current_batch = np.clip(
          np.random.normal(
              current_mean,
              self._params.std_err,
              size=tuple(shape) + current_mean.shape),
          a_min=0,
          a_max=10000)
      series_list.append(current_batch)
      historical_price.pop(0)
      historical_price.append(current_batch[0])   # pick a random sample.

    self._cache = np.array(series_list)
    return self._cache


class GaussianGenerator(TimeSeriesGenerator):
  """Data is sampled from a Gaussian distribution at each step."""

  def __init__(self, init_params, seed):
    """Initialize GaussianGenerator.

    Args:
      init_params: GeneratorParams, initialization parameters.
      seed: to control the randomness of the data generator.
    """
    super().__init__(init_params=init_params, seed=seed)

  def get_next_point(self):
    """Gets the next point in the series."""
    return np.squeeze(
        np.random.normal(self._params.mean, self._params.std_dev, 1))

  def get_series(self, length, shape = ()):
    return np.random.normal(
        self._params.mean, self._params.std_dev,
        size=(length,) + tuple(shape))


class ClipNormalGenerator(TimeSeriesGenerator):
  """Data is sampled from a clipped normal distribution at each step."""

  def __init__(self, init_params, seed):
    """Initialize ClipNormalGenerator.

    Args:
      init_params: GeneratorParams, initialization parameters.
      seed: to control the randomness of the data generator.
    """
    super().__init__(init_params=init_params, seed=seed)

  def get_next_point(self):
    """Gets the next point in the series."""
    return np.clip(np.squeeze(
        np.random.normal(self._params.mean, self._params.std_dev, 1)),
                   self._params.low, self._params.high)

  def get_series(self, length, shape = ()):
    return np.clip(
        np.random.normal(
            self._params.mean, self._params.std_dev,
            size=(length,) + tuple(shape)),
        self._params.low, self._params.high)


class UniformGenerator(TimeSeriesGenerator):
  """Data is sampled from a uniform distribution at each step."""

  def __init__(self, init_params, seed):
    """Initialize GaussianGenerator.

    Args:
      init_params: GeneratorParams, initialization parameters.
      seed: to control the randomness of the data generator.
    """
    super().__init__(init_params=init_params, seed=seed)

  def get_next_point(self):
    """Gets the next point in the series."""
    return np.squeeze(
        np.random.uniform(self._params.low, self._params.high, 1))

  def get_series(self, length, shape = ()):
    return np.random.uniform(
        self._params.low, self._params.high, size=(length,) + tuple(shape))


GENERATOR_MAP = immutabledict.immutabledict({
    'ar': AutoRegressiveGenerator,
    'gaussian': GaussianGenerator,
    'clipnormal': ClipNormalGenerator,
    'uniform': UniformGenerator,
})
