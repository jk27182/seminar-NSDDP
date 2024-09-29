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

import datetime
import math
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import cutting_planes_pb2
import sddp_agent
import sddp_env

FLAGS = flags.FLAGS
flags.DEFINE_bool('export_cut_approximation_func', False,
                  'Whether to export cut approximation function for training.')
flags.DEFINE_bool(
    'init_cut_approximation_lower_bound', False,
    'Whether to randomly initialize cut_approximation_lower_bound'
    'with an initial backward pass over randomly generated'
    'feasible solutions.')
flags.DEFINE_float('demand_mean', 10.0, 'Mean value of the demand.')
flags.DEFINE_float('demand_std_dev', 0.001, 'Standard deviation of the demand.')
flags.DEFINE_float('min_demand', 0.001, 'Lower bound of the demand.')
flags.DEFINE_float('max_demand', 100, 'Max value of the demand.')
flags.DEFINE_float('min_price', 0.001, 'Lower bound of the price.')
flags.DEFINE_float('max_price', 100, 'Max value of the price.')
flags.DEFINE_float('min_capacity', 0.01, 'Lower bound of the capacity.')
flags.DEFINE_float('max_capacity', 1000, 'Max value of the capacity.')

flags.DEFINE_float('sale_price_mean', 10.0,
                   'Sales price mean. Negative value for sales as cost.')
flags.DEFINE_float('sale_price_std_dev', 0.001,
                   'Standard deviation of the sales price.')
flags.DEFINE_float('supply_capacity_mean', 1000,
                   'Capacity mean from a supplier.')
flags.DEFINE_float('supply_capacity_std_dev', 0.001,
                   'Capacity std dev from a supplier.')
flags.DEFINE_integer('num_supplier', 1, 'Num of suppliers.')
flags.DEFINE_integer('num_customer', 1, 'Num of customers.')
flags.DEFINE_integer('num_inventory', 1, 'Num of inventories.')
flags.DEFINE_float('procurement_price_mean', 4.0,
                   'Mean value of the procurement price.')
flags.DEFINE_float('procurement_price_std_dev', 0.001,
                   'Standard deviation of the procurement price.')
flags.DEFINE_float('transportation_price_mean', 0.0,
                   'Mean value of the procurement price.')
flags.DEFINE_float('transportation_price_std_dev', 0.001,
                   'Standard deviation of the procurement price.')
flags.DEFINE_float('max_inventory', 200, 'Max capacity of the inventory.')
flags.DEFINE_float('init_inventory', 10, 'Initial level of the inventory.')
flags.DEFINE_float('holding_cost', 0.01, 'Holding cost per unit inventory.')
flags.DEFINE_float('transport_cost_mean', 0.5,
                   'Unit sales transport cost mean.')
flags.DEFINE_float('transport_cost_std_dev', 3,
                   'Unit sales transport cost standard dev.')
flags.DEFINE_float('transport_cost_max', 1,
                   'Unit sales transport cost max value.')

flags.DEFINE_integer('horizon_len', '5', 'Horizon length to plan.')
flags.DEFINE_integer('batch_size', '16', 'Batch size.')
flags.DEFINE_integer('num_iterations', '100', 'Num of iterations.')
flags.DEFINE_float('convergence_gap', '0.1',
                   'Gap between upper and lower bound for convergence.')
flags.DEFINE_float('confidence_interval_gap', '0.1',
                   'Upper bound confidence interval convergence gap.')

flags.DEFINE_float('init_q_lower_bound', -20000,
                   'Max capacity of the inventory.')

flags.DEFINE_enum('demand_generator_name', 'gaussian', [
    'gaussian', 'clipnormal', 'uniform'], 'Name of the demand generator')

flags.DEFINE_integer('export_interval', '10', 'Num of steps between export.')
flags.DEFINE_bool(
    'use_standard_lp_tf_op', True,
    'Use standard SDDP LP formulation with equality constraints.')

_Z_VALUE_AT_95 = 1.96  # z_{1-alpha/2} = 1.96 when alpha = 0.05


def _compute_95_confidence_interval(mean_value, sample_variance, sample_size):
  # STOCHASTIC DUAL DYNAMIC PROGRAMMING - A REVIEW
  # http://www.optimization-online.org/DB_FILE/2021/01/8217.pdf
  # P11, (1.17).
  interval = _Z_VALUE_AT_95 * math.sqrt(sample_variance / sample_size)
  return mean_value - interval, mean_value + interval


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export_file_handler = None
  param_gen, stage0_solution = sddp_env.get_environment(FLAGS)

  system_config = cutting_planes_pb2.SystemConfig(
      sale_price_mean=FLAGS.sale_price_mean,
      sale_price_std_dev=FLAGS.sale_price_std_dev,
      procurement_price_mean=FLAGS.procurement_price_mean,
      procurement_price_std_dev=FLAGS.procurement_price_std_dev,
      holding_cost=FLAGS.holding_cost,
      demand_mean=FLAGS.demand_mean,
      demand_std_dev=FLAGS.demand_std_dev,
      supply_capacity_mean=FLAGS.supply_capacity_mean,
      supply_capacity_std_dev=FLAGS.supply_capacity_std_dev,
      max_inventory=FLAGS.max_inventory,
      init_inventory=FLAGS.init_inventory,
      num_supplier=FLAGS.num_supplier,
      num_customer=FLAGS.num_customer,
      num_inventory=FLAGS.num_inventory)

  agent = sddp_agent.SDDPAgent(
      system_config=system_config,
      param_gen=param_gen,
      num_stages=FLAGS.horizon_len,
      num_scenarios_per_stage=FLAGS.batch_size,
      max_num_iterations=10,
      init_q_lower_bound=FLAGS.init_q_lower_bound,
      stage0_solution=stage0_solution,
      use_standard_lp_tf_op=FLAGS.use_standard_lp_tf_op,
      export_file_handler=export_file_handler)

  if FLAGS.init_cut_approximation_lower_bound:
    agent.init_cut_approximation_lower_bound()

  start_time = datetime.datetime.now()
  for iteration in range(FLAGS.num_iterations):
    scenario_path_map, total_cost_upper_bound_mean, total_cost_upper_bound_variance, stage_costs = agent.forward_pass(
        iteration)
    # logging.info(scenario_path_map)
    total_cost_lower_bound = agent.backward_pass(iteration, scenario_path_map)
    logging.info('cost bounds [%f, %f], upper_bound_std_dev: %f',
                 total_cost_lower_bound, total_cost_upper_bound_mean,
                 math.sqrt(total_cost_upper_bound_variance))

    if total_cost_upper_bound_mean - total_cost_lower_bound <= FLAGS.convergence_gap:
      training_time = datetime.datetime.now() - start_time
      logging.info('upper_bound_mean_to_lower_bound_converge_time: %f',
                   training_time.total_seconds())
    (upper_bound_confidence_interval_lower,
     upper_bound_confidence_interval_upper) = _compute_95_confidence_interval(
         total_cost_upper_bound_mean, total_cost_upper_bound_variance,
         FLAGS.batch_size)

    if (upper_bound_confidence_interval_upper -
        upper_bound_confidence_interval_lower <= FLAGS.confidence_interval_gap
        and total_cost_lower_bound > upper_bound_confidence_interval_lower):
      training_time = datetime.datetime.now() - start_time
      logging.info('upper_bound_ci_lower_to_lower_bound_converge_time: %f',
                   training_time.total_seconds())

    if iteration % FLAGS.export_interval == 0:
      agent.export_cut_approximate_funcs()

if __name__ == '__main__':
  app.run(main)
