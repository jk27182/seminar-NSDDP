import pickle

import model_solver
import utils
import jax
import jax.numpy as jnp
import neural_sddp.piecewise_nn.cond_piecewise_nn as cpn


n_pieces = 2
n_vars = 1
seed = 1
n_epochs = 10_000_000
tolerance = 1e-4
train_portion = 0.8 * (1/2)**5
learning_rate = 0.0004

key = jax.random.PRNGKey(seed)

model = cpn.CondPiecewiseNN(
    num_vars=n_vars,
    num_pieces=n_pieces,
    num_layers=1,
    num_stages=2,
    hidden_size=128,
)

date_time_str = utils.get_date_time_string()
with open("model_params_" + date_time_str, "rb") as f:
    params = pickle.load(f)

with open("targets", "rb") as f:
    targets = pickle.load(f)

with open("features", "rb") as f:
    features = pickle.load(f)

print(len(features))
for feature in features:
    stage = 1
    print(feature)
    print(feature.shape)
    feature = jnp.reshape(feature, (4, 10), order="F").T
    print(feature)
    print(1/0)
    cuts = model.apply(params, feature, stage)

    problem_wo_cuts = model_solver.construct_test_problem(features=feature)
    problem_w_cuts = model_solver.construct_test_problem(features=feature, cuts=cuts)

    model_solver.solve_msp(problem_w_cuts)
    model_solver.solve_msp(problem_w_cuts)