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

"""Tensorflow based implementation of SDDP."""

import enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from absl import logging
import attr
import canonical_lp_converters
import cutting_planes_pb2
import numpy as np
import recordio
import tensorflow as tf
import tf_ortools_mod   # Dependency on internal proprietary code.
import time_series_generator


@attr.s(auto_attribs=True)
class LinearOptimizationParams:
  """Represents parameters of a batch of the linear optimizations in SDDP.

  The optimization problem is of form:
  minimize c_vector * x_t
  subject to:
    A_matrix * x_t + B_matrix * x_{t-1} <= b_vector
  where c_vector and b_vector are dependent on sampled scenarios, and A_matrix
  and B_matrix are intrinsic to the dynamic system, independent of scenarios.
  """
  # A_matrix shape (num_constraints, num_vars)
  A_matrix: np.ndarray
  # B_matrix shape (num_constraints, num_vars)
  B_matrix: np.ndarray
  # b_vector shape (batch_size, num_constraints,)
  b_vector: np.ndarray
  # c_vector shape (batch_size, num_vars,)
  c_vector: np.ndarray

  def debug_string(self):
    return (f'A_matrix {np.array2string(self.A_matrix)} \n'
            f'B_matrix {np.array2string(self.B_matrix)} \n'
            f'b_vector {np.array2string(self.b_vector)} \n'
            f'c_vector {np.array2string(self.c_vector)} \n')


@enum.unique
class ObjectiveType(enum.Enum):
  """Type of optimization objective."""
  MAX = 'maximization'
  MIN = 'minimization'


@attr.s(auto_attribs=True)
class CuttingPlane:
  """Represents a cutting plane.

  A cutting plane is a hyperplane represented as l(x) = intercept + gradient * x
  """
  # The function value where the cutting plane intercept with
  # y-axis (the function value axis).
  intercept: float
  # The subgradient of the original function evaluated at the point where
  # the cutting plane intersects the orginal function.
  gradient: np.ndarray


class CutApproximateFunc:
  """Represents an approximation function based on a list of cutting planes."""

  def __init__(self, num_vars, lower_bound):
    self._num_vars = num_vars
    self._cuts = [CuttingPlane(intercept=lower_bound,
                               gradient=np.zeros(self._num_vars))]

  def add_cut(self, cutting_plane):
    self._cuts.append(cutting_plane)

  def get_all_cuts(self):
    return self._cuts

  def debug_string(self):
    cutting_plane_format = 'intercept: {0}, gradient: {1}'
    return ''.join([
        cutting_plane_format.format(cut.intercept,
                                    np.array2string(cut.gradient))
        for cut in self._cuts
    ])


class SDDPAgent():
  """Entity which execute SDDP algorithm over a set of sampled scenarios.

  We take the notations from A. Shaprio,
  "Analysis of Stochastic Dual Dynamic Programming Method", 2011:

  * [x(t), t = 1, ..., T], is a n-dimensional vector, indicating the decision
    variables at stage t.
  * For t = 1, ..., T-1:
        Q(x(t), xi(t)) = min_{x(t)} c(t) x(t) + E_{xi(t+1)}[Q(x(t+1), xi(t+1))]
    For t = T:
        Q(x(T), xi(T)) = min_{x(T)} c(T)x(T)
    where x(t) subject to
      * B(t) x(t-1) + A(t) x(t) <= b(t), for t = 2, ..., T
      * A(t) x(t) <= b(t), for t = 1
      * x(t) >= 0.
    and xi(t) is an external random process.
  """

  def __init__(
      self,
      system_config,
      param_gen,
      num_stages,
      num_scenarios_per_stage,
      max_num_iterations,
      init_q_lower_bound,
      stage0_solution,
      use_standard_lp_tf_op = True,
      export_file_handler = None):
    """Initialize a SDDP agent.

    Args:
      system_config: system config.
      param_gen: static values and parameter generators for the stage-wise
        linear optimization problem.
      num_stages: number of stages.
      num_scenarios_per_stage: number of scenarios to be generated in one stage.
      max_num_iterations: maximum number of iterations of the algorithm. Note
        that the algorithm may end earlier if convergence criteria is met.
      init_q_lower_bound: lower bound of the cost to go function.
      stage0_solution: solution of stage 0 to initialize the system.
      use_standard_lp_tf_op: if true, use glop_sddp_lp, where slack variables
        are introduced to SDDP formulation to make the equality constraints
         before calling the LP solver; otherwise use glop_relax_sddp_lp.
         Experiments show no difference in the dual variables associated with
         the equality (with slack vars) vs. nonequality <= constraints and thus
         no difference in the cutting plane generation in the backward pass.
         The running time may be different: glop_sddp_lp with equality
         constraint only in LP solver seems about 5X faster.
      export_file_handler: export training points. For SDDP performance
        benchmark, set to None, no export will be performed.
    """
    self.export_file_handler = export_file_handler
    self.param_gen = param_gen
    self.system_config = system_config
    self.num_vars = param_gen.num_vars
    self.num_constraints = param_gen.num_constraints
    self.num_stages = num_stages
    self.num_scenarios_per_stage = num_scenarios_per_stage
    self._max_num_iterations = max_num_iterations
    self._use_standard_lp_tf_op = use_standard_lp_tf_op
    self.init_q_lower_bound = init_q_lower_bound
    self.stage0_solution = stage0_solution

    # solution_dict: key: stage (starting from 0 to self.num_stages)
    #                value: ndarray w/shape (num_scenarios_per_stage, num_vars).
    self.solution_dict = self._init_empty_solution_dict()

    # Dict of cutting plane approximiations of the cost-to-go function:
    # key: stage \in [2, T+1],
    # value: mathfrak{Q}_{t+1}(x) = E_{xi(t+1)}[Q(x(t+1), xi(t+1))].
    # t = 1, ..., T. Note that mathfrak{Q}_{T+1} = 0 by definition.
    self.cut_approx_cost_to_go_funcs = {
        stage: CutApproximateFunc(self.num_vars, self.init_q_lower_bound)
        for stage in range(2, self.num_stages + 1)
    }
    self.cut_approx_cost_to_go_funcs[self.num_stages + 1] = CutApproximateFunc(
        self.num_vars, 0)

  def _init_empty_solution_dict(self):
    """Initialize an empty solution dict.

    Returns:
      solution_dict: key: stage (starting from 0 to self.num_stages)
        value: a batch of trial solutions in ndarray with shape
          (num_scenarios_per_stage, num_vars).
    """
    # Note that solution at stage = 0 is only initialized for recursive
    # computation convenience and has no real meaning.
    solution_dict = {
        stage: np.zeros((self.num_scenarios_per_stage, self.num_vars))
        for stage in range(self.num_stages + 1)
    }
    solution_dict[0] = np.repeat(np.expand_dims(self.stage0_solution, axis=0),
                                 self.num_scenarios_per_stage, axis=0)
    return solution_dict

  def _sample_scenarios(self):
    """Samples and generates a list of LinearOptimizationParams.

    Returns:
      param_map: key: stage, value: a single stage batch
        LinearOptimizationParams for all the sampled scenarios.
    """

    b_vector_list = []
    for b_gen in self.param_gen.b_gen_list:
      if isinstance(b_gen, time_series_generator.TimeSeriesGenerator):
        # Sample a batch vector with shape (num_stages, num_scenarios_per_stage)
        b_vector = b_gen.get_series(
            length=self.num_stages, shape=(self.num_scenarios_per_stage,))
      else:
        # Duplicate the static parameter into a vector with shape:
        # (num_stages, num_scenarios_per_stage)
        b_vector = np.tile([b_gen],
                           (self.num_stages, self.num_scenarios_per_stage))
      b_vector_list.append(b_vector)
    # b_vectors shape (num_stages, num_scenarios_per_stage, num_constraints)
    b_vectors = np.transpose(np.array(b_vector_list), (1, 2, 0))

    c_vector_list = []
    for c_gen in self.param_gen.c_gen_list:
      if isinstance(c_gen, time_series_generator.TimeSeriesGenerator):
        # Sample a batch vector with shape (num_stages, num_scenarios_per_stage)
        c_vector = c_gen.get_series(
            length=self.num_stages, shape=(self.num_scenarios_per_stage,))
      else:
        # Duplicate the static parameter into a vector with shape:
        # (num_stages, num_scenarios_per_stage)
        c_vector = np.tile([c_gen],
                           (self.num_stages, self.num_scenarios_per_stage))
      c_vector_list.append(c_vector)
    # c_vectors shape (num_stages, num_scenarios_per_stage, num_vars)
    c_vectors = np.transpose(np.array(c_vector_list), (1, 2, 0))

    param_map = {
        # Key is stage, which start from 1.
        index + 1:
        LinearOptimizationParams(self.param_gen.A_matrix,
                                 self.param_gen.B_matrix,
                                 b_vectors[index, :, :], c_vectors[index, :, :])
        for index in range(self.num_stages)
    }
    return param_map

  def forward_pass(
      self, iteration
  ):
    """Run a forward pass of SDDP algorithm.

      Forward pass samples scenario paths and computes solution_dict recursively
      from stage 1 to stage T, based on sampled scenario paths and
      self.cut_approx_cost_to_go_funcs.
    Args:
      iteration: iteration # for this forward pass.

    Returns:
      scenario_path_map: key: stage, value: a single stage batch
        LinearOptimizationParams for all the sampled scenarios.
      solution_dict: key: stage, value a batch of trial solutions in ndarray.
    """
    # scenario_path_map: a list of batch LP params of length num_stages.
    scenario_path_map = self._sample_scenarios()

    total_cost_upper_bound = 0
    total_cost_variance = 0
    stage_cost_list = []
    for stage in range(1, self.num_stages + 1):
      logging.info('forward_pass @iteration: %d; stage %d', iteration, stage)

      self.solution_dict[stage], _, _, _ = self.solve_single_stage(
          self.solution_dict[stage - 1],
          self.cut_approx_cost_to_go_funcs[stage + 1],
          # scenario_path_map is a list starting from index 0.
          # stage starts from 1.
          scenario_path_map[stage])
      # Gets a single stage cost based on the cost func: c(t) x(t)
      # cost shape (batch, num_vars)
      # current_stage_solution shape (batch, num_vars)
      stage_cost = np.diagonal(np.inner(self.solution_dict[stage],
                                        scenario_path_map[stage].c_vector))

      stage_cost_expectation = np.mean(stage_cost, axis=0)
      # Trial solution in an iteration provides a feasible solution for the
      # original problem and aggregating the stage-wise cost expectation derived
      # from trial solutions over all stages gives an upper bound for the total
      # cost.
      total_cost_upper_bound += stage_cost_expectation
      optimal_cost_variance = np.var(stage_cost, axis=0)
      # Assuming stage-wise costs are indepedent random variables.
      # The variance of their sum is the same as the sum of their variances.
      total_cost_variance += optimal_cost_variance
      stage_cost_list.append(stage_cost_expectation)

      logging.info('stage_cost_expectation%d: %f',
                   stage, stage_cost_expectation)
      logging.info('stage_cost_variance: %f', optimal_cost_variance)

    return scenario_path_map, total_cost_upper_bound, total_cost_variance, stage_cost_list

  def solve_single_stage(
      self, last_stage_solution,
      next_stage_cut_approx_cost_to_go_func,
      scenario_params
  ):
    """Solve a single stage problem based on the approximate cost to go func.

    The problem at stage t is defined as:
      min_{x(t), y(t)}: c(t) x(t) + y(t)
    where x(t) subject to
      * B(t) x(t-1) + A(t) x(t) <= b(t), for t = 2, ..., T
      * A(t) x(t) <= b(t), for t = 1
      * x(t) >= 0
    y(t) is the approximate cost-to-go func based on cutting-planes.
      y(t) = max{all cuts}. Let the ith cutting planes be represented by
        (intercept_i, gradient_i), then
      y(t) >= intercept_i(t+1) + gradient_i(t+1) (x_t) for i = 1, ..., num_cuts.
      let intercepts be the vector of shape (num_cuts, ) and gradients be
      the vector of shape (num_cuts, num_vars)
      gradient(t+1) (x_t) - y(t) <= -intercepts

    Given t, we have the following LP problem in vector form for each scenario:
      * variable [x, y], where shape of x: (num_vars,) y is a scalar.
      * cost coefficient c with shape (num_vars,)
      * constraint coefficient matrix A with shape (num_constraints, num_vars),
      * constraint upper bound vector b - B * last_stage_x with shape
        (num_constraints, )
      * cut gradient matrix g with shape (num_cuts, num_vars)
      * cut intercept vector p with shape (num_cuts,)

    Args:
      last_stage_solution: solution from the stage t-1: x_{t-1}
      next_stage_cut_approx_cost_to_go_func: approximate cost-to-go function of
        stage t+1: mathfrak{Q}_{t+1}(x_t).
      scenario_params: batch LP parameters at stage t.

    Returns:
      opt_sol: batch optimal solution for x only.
      dual: dual variable from all constraints.
      opt_obj_value: optimal objective value.
    """
    # cost shape (batch, num_vars)
    cost = scenario_params.c_vector

    # Initialize coefficient as [A, 0] with shape
    # (batch, num_constraints, num_vars)
    coefficient = np.repeat(
        np.expand_dims(
            # scenario_params.A_matrix shape (num_constraints, num_vars)
            scenario_params.A_matrix,
            axis=0),
        self.num_scenarios_per_stage,
        axis=0)

    # Initialize upper_bound as [b - B * last_stage_x]
    # shape (batch, num_constraints)
    # scenario_params.b_vector shape (batch, num_constraints)
    # last_stage_solution shape (batch, num_vars)
    # scenario_params.B_matrix shape (num_constraints, num_vars)
    upper_bound = scenario_params.b_vector - np.matmul(
        last_stage_solution, np.transpose(scenario_params.B_matrix))

    all_cuts = next_stage_cut_approx_cost_to_go_func.get_all_cuts()
    # intercepts shape (batch, num_cuts)
    intercepts = np.repeat(
        # cut.intercept is a scalar
        np.expand_dims([cut.intercept for cut in all_cuts], axis=0),
        self.num_scenarios_per_stage,
        axis=0)
    # gradients shape (batch, num_cuts, num_vars)
    # cut.gradient shape [num_vars]
    gradients = np.repeat(
        np.expand_dims([cut.gradient for cut in all_cuts], axis=0),
        self.num_scenarios_per_stage,
        axis=0)

    # opt_sol.shape: (batch, num_vars +1)
    # dual.shape: (batch, num_constraints + num_cuts)
    # status.shape: (batch,)
    # opt_obj_value.shape: (batch,)

    # Solve the LP problem in batch via an internal TF operator.
    if self._use_standard_lp_tf_op:
      opt_sol, q_bound, dual_constraints, dual_cuts, status, opt_obj_value = tf_ortools_mod.glop_sddp_lp(
          c=cost, a=coefficient, b=upper_bound, g=gradients, p=intercepts)
    else:
      opt_sol, q_bound, dual_constraints, dual_cuts, status, opt_obj_value = tf_ortools_mod.glop_relax_sddp_lp(
          c=cost, a=coefficient, b=upper_bound, g=gradients, p=intercepts)
    if tf.reduce_any(tf.cast(status, dtype=tf.bool)):
      raise ValueError('opt status should be 0.')

    # only take the x part of the solution.
    return opt_sol, dual_constraints, dual_cuts, opt_obj_value

  def backward_pass(self, iteration,
                    scenario_path_map):
    """Run a backward pass of SDDP algorithm.

      Backward updates self.cut_approx_cost_to_go_funcs recursively
      from stage T-1 to stage 1, based on the given scenario paths.
    Args:
      iteration: iteration # for this forward pass.
      scenario_path_map: sampled scenario path map from forward pass, keyed by
        stage.
    Returns:
      total_cost_lower_bound: lower bound estimated from first stage cost.
    """
    for stage in range(self.num_stages, 1, -1):
      logging.info('backward_pass @iteration: %d; stage %d', iteration, stage)

      params = scenario_path_map[stage]
      _, dual_constraints, dual_cuts, _ = self.solve_single_stage(
          self.solution_dict[stage - 1],
          self.cut_approx_cost_to_go_funcs[stage + 1], params)
      self.update_cut_approximate(
          stage, np.concatenate([dual_constraints, dual_cuts],
                                axis=1), params.B_matrix, params.b_vector,
          self.cut_approx_cost_to_go_funcs[stage + 1])

    # total_cost_lower_bound is the opt_obj_value of the first stage cost.
    _, _, _, final_opt_obj_value = self.solve_single_stage(
        self.solution_dict[0],
        self.cut_approx_cost_to_go_funcs[2], params)

    total_cost_lower_bound = np.mean(final_opt_obj_value, axis=0)
    return total_cost_lower_bound


  def update_cut_approximate(self, stage, dual_var, B_matrix, b_vector,
                             next_stage_cut_approx_Q):
    """Update the cutting plane approximiation of cost-to-go functions.

    The updated cut is formed by:
    gradient = expectation_{samples} [-B_matrix * dual_var[original_constraints]]

    * if there exist cutting planes in the approximation function.
      intercept = expectation_{samples} [
        b_vector * dual_var[original_constraints] +
        intercepts_next_stage_cut_approx_Q *dual_var[cutting_plane_constraints]]
      Collectively,
      intercept = expectation_{samples} [
        [b_vector, intercepts_next_stage_cut_approx_Q] * dual_var]

    * if there exists no cutting planes in the approximation function,
        e.g., first iteration
      intercept = expectation_{samples}[b_vector*dual_var[original_constraints]]
    combining the two cases above,
      intercept = expectation_{samples} [
        [b_vector, intercepts_next_stage_cut_approx_Q] * dual_var]

    Args:
      stage:
      dual_var: np.ndarray shape (batch, num_constraints + num_cuts)
      B_matrix: np.ndarray shape (num_constraints, num_vars)
      b_vector: np.ndarray shape (batch, num_constraints)
      next_stage_cut_approx_Q: next stage cut approximation function of Q.
    """
    # gradient shape (num_vars, )
    gradient = -np.average(
        # batch gradient shape (batch, num_vars)
        np.matmul(dual_var[:, :self.num_constraints], B_matrix),
        axis=0)

    all_cuts = next_stage_cut_approx_Q.get_all_cuts()
    # np.array(intercepts_next_stage_cut_approx_Q)
    concat_b_intercept_vector = b_vector
    if all_cuts:
      intercepts_next_stage_cut_approx_Q = np.repeat(
          np.expand_dims([cut.intercept for cut in all_cuts], axis=0),
          self.num_scenarios_per_stage,
          axis=0)
      concat_b_intercept_vector = np.concatenate(
          [b_vector, np.array(intercepts_next_stage_cut_approx_Q)],
          axis=1)
    # intercept is a scalar.
    intercept = np.average(
        # concat_b_intercept_vector and dual_var have the same shape
        # (batch, num_constraint+num_cuts)
        np.diag(np.inner(concat_b_intercept_vector, dual_var)))
    self.cut_approx_cost_to_go_funcs[stage].add_cut(
        CuttingPlane(intercept=intercept, gradient=gradient))


  def init_cut_approximation_lower_bound(self):
    """Initialize cut approximation function with random feasible solution."""
    logging.info('init_cut_approximation_lower_bound')
    scenario_path_map = self._sample_scenarios()
    self.solution_dict = {
        stage: np.repeat(
            np.expand_dims(self.stage0_solution, axis=0),
            self.num_scenarios_per_stage,
            axis=0) for stage in range(self.num_stages)
    }
    self.backward_pass(0, scenario_path_map)
    # backward_pass should only change self.cut_approx_cost_to_go_funcs.
    # Reset self.solution_dict for forward pass.
    self._init_empty_solution_dict()

  def get_cut_approximate_funcs(self):
    """Get cut approximate cost-to-go function."""
    list_qfunc = []
    for stage in range(2, self.num_stages + 1):
      all_cuts = self.cut_approx_cost_to_go_funcs[stage].get_all_cuts()
      cut_approximate_func_proto = cutting_planes_pb2.CutApproximateFuncProto()
      for cutting_plane in all_cuts:
        cutting_plane_proto = cutting_planes_pb2.CuttingPlaneProto(
            intercept=cutting_plane.intercept, gradient=cutting_plane.gradient)
        cut_approximate_func_proto.cutting_planes.append(cutting_plane_proto)

      training_point = cutting_planes_pb2.CutApproximateFuncTrainingData(
          cut_approx_cost_to_go_func=cut_approximate_func_proto,
          stage=stage,
          system_config=self.system_config)
      list_qfunc.append(training_point)
    return list_qfunc

  def export_cut_approximate_funcs(self):
    """Export cut approximate cost-to-go function for training."""
    list_qfunc = self.get_cut_approximate_funcs()
    if self.export_file_handler:
      for training_point in list_qfunc:
        self.export_file_handler.WriteRecord(training_point.SerializeToString())
