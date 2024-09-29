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

"""Utils to convert problem specific formulation to canonical LPs."""

from typing import List, Tuple, Union

from absl import logging
import attr
import numpy as np
import time_series_generator


@attr.s(auto_attribs=True)
class CustomerProfile:
  demand: time_series_generator.TimeSeriesGenerator
  max_demand: float
  price: time_series_generator.TimeSeriesGenerator  # price per unit demand.


@attr.s(auto_attribs=True)
class InventoryProfile:
  init_inventory: float
  max_inventory: float  # storage capacity bounds.
  holding_cost: float  # cost per unit inventory.


@attr.s(auto_attribs=True, frozen=True)
class SupplierProfile:
  supplier_outflow_bound: time_series_generator.TimeSeriesGenerator
  price: time_series_generator.TimeSeriesGenerator  # price per unit demand.
  transportation_price: time_series_generator.TimeSeriesGenerator


@attr.s(auto_attribs=True)
class TransportCostProfile:
  mean: float
  std_dev: float
  max: float



class LinearOptimizationParamGenerator:
  """Parameters in canonical T-stage stochastic LP problem.

  For t = 1, ..., T-1:
      Q(x(t), xi(t)) = min_{x(t)} c(t)'x(t) + E_{ xi(t+1) [Q(x(t+1), xi(t+1))]
  For t = T:
      Q(x(T), xi(T)) = min_{x(T)} c(T)' x(T)
  x(t) subject to B(t) x(t-1) + A(t) x(t) <= b(t), x(t) >= 0.
  """

  def __init__(self, A_matrix, B_matrix,
               b_gen_list,
               c_gen_list):
    """Creates a linear program parameters for canonical formulation.

    Args:
      A_matrix: Coefficient for x(t) with shape (num_constraints, num_vars)
      B_matrix: Coefficient for x(t-1) with shape (num_constraints, num_vars)
      b_gen_list: List of constraint constant time series generators, with len
        (num_constraints)
      c_gen_list: List of cost time series generators with len (num_vars)
    """
    self.num_constraints = A_matrix.shape[0]
    self.num_vars = A_matrix.shape[1]

    if self.num_constraints != B_matrix.shape[0]:
      raise ValueError(
          f'Shape mismatch: num_constraints in A_matrix: {A_matrix.shape[0]}'
          ' should be the same as in B_matrix: {B_matrix.shape[0]}')

    if self.num_constraints != len(b_gen_list):
      raise ValueError(
          f'Shape mismatch: num_constraints in A_matrix: {A_matrix.shape[0]}'
          ' should be the same as in b_gen_list: {len(b_gen_list)}')

    if self.num_vars != B_matrix.shape[1]:
      raise ValueError(
          f'Shape mismatch: num_vars in A_matrix: {A_matrix.shape[1]}'
          ' should be the same as in B_matrix: {B_matrix.shape[1]}')

    if self.num_vars != len(c_gen_list):
      raise ValueError(
          f'Shape mismatch: num_vars in A_matrix: {A_matrix.shape[1]}'
          ' should be the same as in c_gen_list: {len(c_gen_list)}')

    self.A_matrix = A_matrix
    self.B_matrix = B_matrix
    self.b_gen_list = b_gen_list
    self.c_gen_list = c_gen_list

  def get_num_vars(self):
    return self.num_vars

  def get_num_constraints(self):
    return self.num_constraints

  def debug_string(self):
    return f'A_matrix {np.array2string(self.A_matrix)} \n B_matrix {np.array2string(self.B_matrix)} \n'


def _get_A_matrix(num_suppliers, num_customers, num_inventories):
  """Compute A_matrix."""
  A_11 = np.tile(np.diagflat(np.ones(num_customers)), num_inventories)
  A_1 = np.concatenate([
      A_11,
      np.zeros((num_customers, num_suppliers * num_inventories)),
      np.zeros((num_customers, num_inventories))
  ],
                       axis=1)

  A_22 = np.repeat(np.diagflat(np.ones(num_suppliers)), num_inventories, axis=1)
  A_2 = np.concatenate([
      np.zeros((num_suppliers, num_customers * num_inventories)), A_22,
      np.zeros((num_suppliers, num_inventories))
  ],
                       axis=1)

  A_33 = np.diagflat(np.ones(num_inventories))
  A_3 = np.concatenate([
      np.zeros((num_inventories, num_customers * num_inventories)),
      np.zeros((num_inventories, num_suppliers * num_inventories)),
      A_33,
  ],
                       axis=1)

  A_41 = np.repeat(np.diagflat(np.ones(num_inventories)), num_customers, axis=1)
  A_4 = np.concatenate([
      A_41,
      np.zeros((num_inventories, num_suppliers * num_inventories)),
      np.zeros((num_inventories, num_inventories)),
  ],
                       axis=1)
  A_51 = np.repeat(
      np.diagflat(-np.ones(num_inventories)), num_customers, axis=1)
  A_52 = np.repeat(np.diagflat(np.ones(num_inventories)), num_suppliers, axis=1)
  A_53 = np.diagflat(-np.ones(num_inventories))
  A_5 = np.concatenate([A_51, A_52, A_53], axis=1)
  A_6 = -A_5

  return np.concatenate([A_1, A_2, A_3, A_4, A_5, A_6], axis=0)


def _get_B_matrix(num_suppliers, num_customers, num_inventories):
  """Compute B_matrix."""

  num_all_nodes = num_suppliers + num_customers + num_inventories
  B_1 = np.concatenate([
      np.zeros((num_all_nodes, num_customers * num_inventories)),
      np.zeros((num_all_nodes, num_suppliers * num_inventories)),
      np.zeros((num_all_nodes, num_inventories))
  ],
                       axis=1)

  B_43 = np.diagflat(-np.ones(num_inventories))
  B_4 = np.concatenate([
      np.zeros((num_inventories, num_customers * num_inventories)),
      np.zeros((num_inventories, num_suppliers * num_inventories)),
      B_43,
  ],
                       axis=1)

  return np.concatenate([B_1, B_4, -B_4, B_4], axis=0)


def _get_portfolio_optimization_A_matrix(num_ticker, stock_profiles):
  """Compute A_matrix."""
  id_matrix = np.diagflat(np.ones(num_ticker))

  A_1 = [
      id_matrix,
      np.zeros((num_ticker, num_ticker)),
      np.zeros((num_ticker, num_ticker + 1)),
  ]
  A_2 = [
      np.zeros((1, num_ticker)),
      stock_profiles.procurement_price,
      np.zeros((1, num_ticker + 1)),
  ]
  A_3 = [
      id_matrix,
      -id_matrix,
      id_matrix,
      np.zeros((num_ticker, 1)),
  ]
  A_4 = [
      -id_matrix,
      id_matrix,
      -id_matrix,
      np.zeros((num_ticker, 1)),
  ]
  A_5 = [
      stock_profiles.sales_price,
      stock_profiles.negative_procurement_price,
      np.zeros((1, num_ticker)),
      -np.ones((1, 1)),
  ]
  A_6 = [
      stock_profiles.negative_sales_price,
      stock_profiles.procurement_price,
      np.zeros((1, num_ticker)),
      np.ones((1, 1)),
  ]

  return [A_1, A_2, A_3, A_4, A_5, A_6]


def _get_portfolio_optimization_B_matrix(num_ticker):
  """Compute B_matrix."""

  id_matrix = np.diagflat(np.ones(num_ticker))

  B_1 = np.concatenate([
      np.zeros((num_ticker, num_ticker)),
      np.zeros((num_ticker, num_ticker)),
      -id_matrix,
      np.zeros((num_ticker, 1)),
  ],
                       axis=1)

  B_2 = np.concatenate([
      np.zeros((1, num_ticker)),
      np.zeros((1, num_ticker)),
      np.zeros((1, num_ticker)),
      -np.ones((1, 1)),
  ],
                       axis=1)

  B_4 = np.concatenate([
      np.zeros((num_ticker, num_ticker)),
      np.zeros((num_ticker, num_ticker)),
      id_matrix,
      np.zeros((num_ticker, 1)),
  ],
                       axis=1)

  B_5 = np.concatenate([
      np.zeros((1, num_ticker)),
      np.zeros((1, num_ticker)),
      np.zeros((1, num_ticker)),
      np.ones((1, 1)),
  ],
                       axis=1)

  return np.concatenate([B_1, B_2, B_1, B_4, B_5, B_2], axis=0)


def convert_portfolio_optimization(
    stock_profiles,):
  """Converts an portfolio optimization problem to canonical LP formulation."""
  num_tickers = stock_profiles.num_tickers
  A_matrix = _get_portfolio_optimization_A_matrix(num_tickers, stock_profiles)
  B_matrix = _get_portfolio_optimization_B_matrix(num_tickers)
  b_vector = np.zeros(B_matrix.shape[0])  # num_constraints

  c_vector = [
      stock_profiles.negative_sales_price, stock_profiles.procurement_price,
      np.zeros(num_tickers + 1)
  ]

  stage0_solution = np.concatenate([
      np.zeros(num_tickers),
      np.zeros(num_tickers),
      np.zeros(num_tickers),
      np.array([stock_profiles.init_cash])
  ],
                                   axis=0)

  return LinearOptimizationParamGenerator(A_matrix, B_matrix, b_vector,
                                          c_vector), stage0_solution


def convert_inventory_optimization(
    supplier_profiles,
    customer_profiles,
    inventory_profiles,
    transport_cost_profile,
):
  """Converts an inventory optimization problem to canonical LP formulation."""

  num_suppliers = len(supplier_profiles)  # S
  num_customers = len(customer_profiles)  # C
  num_inventories = len(inventory_profiles)  # V
  logging.info('num_suppliers %d, num_inventories %d, num_customers %d',
               num_suppliers, num_inventories, num_customers)

  A_matrix = _get_A_matrix(num_suppliers, num_customers, num_inventories)
  B_matrix = _get_B_matrix(num_suppliers, num_customers, num_inventories)
  d = [customer.demand for customer in customer_profiles]
  u = [supplier.supplier_outflow_bound for supplier in supplier_profiles]
  v = [inventory.max_inventory for inventory in inventory_profiles]
  b_gen_list = d + u + v + 3 * num_inventories * [0]
  transport_cost = np.clip(
      np.random.normal(
          transport_cost_profile.mean,
          transport_cost_profile.std_dev,
          size=(num_inventories, num_customers)), 0, transport_cost_profile.max)

  customer_cost = []
  for i in range(num_inventories):
    for j, customer in enumerate(customer_profiles):
      customer_cost.append(customer.price + transport_cost[i][j])
  c_gen_list = customer_cost + [
      supplier.price for supplier in supplier_profiles
  ] * num_inventories + [
      inventory.holding_cost for inventory in inventory_profiles
  ]

  stage0_solution = np.concatenate([
      np.zeros(num_customers * num_inventories),
      np.zeros(num_suppliers * num_inventories),
      np.array([inventory_p.init_inventory
                for inventory_p in inventory_profiles])], axis=0)

  return LinearOptimizationParamGenerator(A_matrix, B_matrix, b_gen_list,
                                          c_gen_list), stage0_solution



