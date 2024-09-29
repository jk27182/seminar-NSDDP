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

"""SDDP env config."""

import enum

from absl import logging
import attr
import canonical_lp_converters
import numpy as np
import time_series_generator


@enum.unique
class OptimizationProblemType(enum.Enum):
  INVENTORY = 'inventory'
  PORTFOLIO = 'portfolio'


@attr.s(auto_attribs=True, frozen=True)
class StockProfile:
  sales_price: time_series_generator.TimeSeriesGenerator
  procurement_price: time_series_generator.TimeSeriesGenerator
  negative_sales_price: time_series_generator.TimeSeriesGenerator
  negative_procurement_price: time_series_generator.TimeSeriesGenerator
  num_tickers: int
  init_cash: float


def get_environment(system_config, problem_type, generator_name=None):
  if problem_type == OptimizationProblemType.INVENTORY:
    return get_inventory_optimization_environment(system_config, generator_name)
  else:
    return get_portfolio_optimization_environment(system_config)


def get_portfolio_optimization_environment(system_config):
  """Gets the problem setup for portfolio optimization."""
  generator_name = 'auto_regressive_clipnormal'
  num_tickers = len(system_config.sales_ar_intercept_list)
  sales_coefficient_ndarray = np.reshape(
      np.array(system_config.sales_ar_coefficient_list, dtype=np.float32),
      (num_tickers, -1))

  sales_intercept_ndarray = np.array(
      system_config.sales_ar_intercept_list, dtype=np.float32)
  # shape init_point (#tickers, ar_order)
  sales_init_point_ndarray = np.reshape(
      np.array(system_config.sales_ar_init_point_list, dtype=np.float32),
      (num_tickers, system_config.ar_order))

  ticker_std_err_ndarray = np.array(
      system_config.ticker_std_err_list, dtype=np.float32)
  peak_ndarray = np.array(system_config.peak_list, dtype=np.float32)

  generator = time_series_generator.GENERATOR_MAP[generator_name]

  sales_price_gen = generator(
      time_series_generator.GeneratorParams(
          coefficients=sales_coefficient_ndarray,
          intercepts=sales_intercept_ndarray,
          init_point=sales_init_point_ndarray,
          std_err=ticker_std_err_ndarray,
          peak=peak_ndarray),
      seed=1)
  procurement_price_gen = time_series_generator.DeltaTimeSeriesGenerator(
      sales_price_gen)
  negative_sales_price_gen = time_series_generator.NegationTimeSeriesGenerator(
      sales_price_gen)
  negative_procurement_price_gen = time_series_generator.NegationTimeSeriesGenerator(
      procurement_price_gen)

  stock_profiles = StockProfile(
      sales_price=sales_price_gen,
      procurement_price=procurement_price_gen,
      negative_sales_price=negative_sales_price_gen,
      negative_procurement_price=negative_procurement_price_gen,
      num_tickers=num_tickers,
      init_cash=system_config.init_cash)

  logging.info(stock_profiles)

  return canonical_lp_converters.convert_portfolio_optimization(stock_profiles)


def get_inventory_optimization_environment(system_config, generator_name=None):
  """Gets the problem setup for inventory optimization."""
  if generator_name is None:
    generator_name = system_config.demand_generator_name
  generator = time_series_generator.GENERATOR_MAP[generator_name]

  supply_capacity_gen = generator(
      time_series_generator.GeneratorParams(
          low=system_config.min_capacity,
          high=system_config.max_capacity,
          mean=system_config.supply_capacity_mean,
          std_dev=system_config.supply_capacity_std_dev),
      seed=1)
  procurement_price_gen = generator(
      time_series_generator.GeneratorParams(
          low=system_config.min_price,
          high=system_config.max_price,
          mean=system_config.procurement_price_mean,
          std_dev=system_config.procurement_price_std_dev),
      seed=1)
  sales_price_gen = -system_config.sale_price_mean
  # sales_price_gen = generator(
  #    time_series_generator.GeneratorParams(
  #        low=-system_config.max_price, high=-system_config.min_price,
  #        mean=-system_config.sale_price_mean, std_dev=system_config.sale_price_std_dev),
  #    seed=1)
  demand_gen = generator(
      time_series_generator.GeneratorParams(
          low=system_config.min_demand,
          high=system_config.max_demand,
          mean=system_config.demand_mean,
          std_dev=system_config.demand_std_dev),
      seed=1)

  transportation_price_gen = generator(
      time_series_generator.GeneratorParams(
          mean=system_config.transportation_price_mean,
          std_dev=system_config.transportation_price_std_dev),
      seed=1)

  inventory_profile = canonical_lp_convertersInventoryProfile(
      init_inventory=system_config.init_inventory,
      max_inventory=system_config.max_inventory,
      holding_cost=system_config.holding_cost)

  supplier_profile = canonical_lp_convertersSupplierProfile(
      supplier_outflow_bound=supply_capacity_gen,
      price=procurement_price_gen,
      transportation_price=transportation_price_gen)
  customer_profile = canonical_lp_convertersCustomerProfile(
      demand=demand_gen,
      max_demand=system_config.max_demand,
      price=sales_price_gen,
      direct_sale_edges=np.ones((system_config.num_supplier,), dtype=int))
  transport_cost_profile = canonical_lp_converters.TransportCostProfile(
      mean=system_config.transport_cost_mean,
      std_dev=system_config.transport_cost_std_dev,
      max=system_config.transport_cost_max)
  logging.info(supplier_profile)
  logging.info(inventory_profile)
  logging.info(customer_profile)
  logging.info(transport_cost_profile)

  return canonical_lp_converters.convert_inventory_optimization(
      supplier_profiles=[supplier_profile] * system_config.num_supplier,
      customer_profiles=[customer_profile] * system_config.num_customer,
      inventory_profiles=[inventory_profile] * system_config.num_inventory,
      transport_cost_profile=transport_cost_profile)
