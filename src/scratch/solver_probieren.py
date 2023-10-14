import jax
import jax.numpy as jnp
import optax
from absl import flags
import pandas as pd

from msppy.utils.examples import construct_nvid
from msppy.solver import SDDP, Extensive
from msppy.evaluation import Evaluation, EvaluationTrue
from neural_sddp.piecewise_nn.cond_piecewise_nn import CondPiecewiseNN
from neural_sddp.piecewise_nn.cond_piecewise_nn import train_loop

seed = 1
nvm = construct_nvid()
# nvm.discretize(n_Markov_states=2, n_sample_paths=1000, method='SA');
nvm_sddp = SDDP(nvm)
nvm_sddp.solve(freq_evaluations=1, n_simulations=-1, tol=10**(-4), logFile=0)
# nvm.write_cuts('schnitte')
df_dict, df_cuts = nvm.write_cuts_to_df()
num_vars = 1
num_pieces = len(df_cuts)

import pickle
with open("data.pckl", "rb") as f:
    additional_data = pickle.load(f)
eingabe = list(additional_data.keys())
ausgabe = list(additional_data.values())

model = CondPiecewiseNN(
    num_vars=num_vars,
    num_layers=1,
    num_stages=2,
    hidden_size=3,
    num_pieces=num_pieces,
    # num_pieces=len(df_cuts),
)

# flags.DEFINE_string('loss_type', 'emd', 'mse/emd')
key = jax.random.PRNGKey(seed)

# Für unconstraintes u ist inf nicht möglich da dann in NN nur NaNs rauskommen
d = {
    'a':0, 'b':10, 'c':-1,
    'u': 10_000, 'q':2, 'r': 0.5,
    'uncert': 11,
}

def create_input_array(dim_mat, data):
    A = jnp.eye(dim_mat)
    B = jnp.eye(dim_mat)
    # Ist es schneller, ones mit Faktor zu benutzen oder
    # ist es schneller np.tile zu benutzen?
    # Wahrscheinlich np.tile
    c = data['a'] * jnp.ones((dim_mat,1))
    a = data['a'] * jnp.ones((dim_mat,1))
    b = data['b'] * jnp.ones((dim_mat,1))
    u = data['u'] * jnp.ones((dim_mat,1))
    r = data['r'] * jnp.ones((dim_mat,1))
    q = data['q'] * jnp.ones((dim_mat,1))
    uncert = data['uncert']* jnp.ones((dim_mat,1))
    # input_data = jnp.column_stack((A, B, uncert, c, , b, u, r, q)).flatten()
    input_data = jnp.column_stack((uncert, c, u, r, q)).flatten()
    return input_data

input_data = create_input_array(dim_mat=2, data=d)
feature = jnp.array(list(d.values()))
feature = input_data

target = df_cuts.iloc[:num_pieces, :-1].values
# target = jnp.reshape(target, (-1, len(df_cuts), num_vars +1))
target = jnp.reshape(target, (-1, len(target), num_vars +1))
print('target')
print(target)
print('DIE FEATURES')
print(feature)
stage = 1
n_epochs = 10_000_000
tolerance = 1e-4
params = model.init(key, feature, stage)
optimizer = optax.adam(learning_rate=0.4)

params, loss = train_loop(model, params, optimizer, n_epochs, tolerance, feature, stage, target)

print('der loss')
print(loss)
output = model.apply(params, feature, stage)
print('der output')
print(output)
df_out = pd.DataFrame(output[0])
df_out['stage'] = 1
nvm = construct_nvid()
from msppy.msp import MSLP

def construct_problem(u, c, q, r, uncertainty):
    nvid = MSLP(T=2, sense=1, bound=u)
    for t in range(2):
        m = nvid[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=c)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=q)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=r)
            m.addConstr(sold + unsatisfied == 0,
            uncertainty={'rhs':range(uncertainty)})
            m.addConstr(sold + recycled == buy_past)

    return nvid

prob = construct_problem(20, -1.2, 2, 0.5, 11)
# nvm.discretize(n_Markov_states=10, n_sample_paths=1000, method='SA');
# nvm.read_cuts_from_df(df_cuts)
# prob.read_cuts_from_df(df_out)
# nvm.read_cuts('schnitte')
nvm_sddp = SDDP(prob)
nvm_sddp.solve(freq_evaluations=1, n_simulations=-1, tol=10**(-4))
# print(len(df_dict))
# for key, val in df_dict.items():
#     print(key)
#     print(val)
print('nvm_sddp.db[-1]',nvm_sddp.db[-1])
print('nvm_sddp.first_stage_solution',nvm_sddp.first_stage_solution)
# res = Evaluation(nvm)
# res.run(n_simulations=-1)
# print('res.gap', res.gap)
# res_true = EvaluationTrue(nvm)
# res_true.run(n_simulations=3000)
# print('res_true.CI', res_true.CI)
# #print('res_true.pv',res_true.pv)
# import numpy as np
# print(np.mean(res_true.pv))
# print('res_true.pv',len(res_true.pv))
# print(res_true.n_sample_paths)
# print(res_true.n_simulations)
