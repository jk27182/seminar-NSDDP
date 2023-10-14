from msppy.msp import MSLP
import numpy as np
from msppy.solver import SDDP
from msppy.evaluation import Evaluation, EvaluationTrue

import utils as utils
from neural_sddp.piecewise_nn.cond_piecewise_nn import CondPiecewiseNN


samples = 1
T = 3
means = [5, 2, -1.0, 20, 2, 0.5]
stds = [2, 1.5, 2, 10, 1, .25]
stds = [0 for _ in means]
arr_mean = np.random.normal(means[0], stds[0], size=(samples,))
arr_std = np.random.normal(means[1], stds[1], size=(samples,))
arr_std = np.clip(arr_std, 0.1, 10)
arr_c = np.random.normal(means[2], stds[2], size=(samples,))
arr_u = np.random.normal(means[3], stds[3], size=(samples,))
arr_q = np.random.normal(means[4], stds[4], size=(samples,))
arr_r = np.random.normal(means[5],stds[5], size=(samples,))


def create_data(A,B,b,c,u,q,r,uncert, deviation_prct):
    # Brauch man das überhaupt?
    arr_A = np.random.normal(A, a*deviation_prct)
    arr_B = np.random.normal(B, a*deviation_prct)
    # Wo ist W?
    # Haben die das vom Paper vllt schon gemacht?
    arr_b = np.random.normal(b, b*deviation_prct)
    arr_c = np.random.normal(c, c*deviation_prct)
    arr_u = np.random.normal(u, u*deviation_prct)
    arr_q = np.random.normal(q, q*deviation_prct)
    arr_r = np.random.normal(r, r*deviation_prct)
    arr_uncert = np.random.normal(u, uncert*deviation_prct)


max_iter_solv = 100
data_all = []
# Create coefficients
for i in range(samples):
    d = {
        'T': 3, 'mean': arr_mean[i], 
        'std':arr_std[i], 'c':arr_c[i],
        'u':arr_u[i], 'q':arr_q[i], 'r': arr_r[i],
    }
    data_all.append(d)


def f(random_state):
    return random_state.lognormal(mean=np.log(4),sigma=2)

def f2(random_state):
    return random_state.normal(loc=data['mean'], scale=data['std'])

# Create Models with coefficients from above
model_list = []
for data in data_all:
    nvic = MSLP(T=data['T'], sense=-1, bound=100)
    for t in range(data['T']):
        m = nvic[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=data['c'], ub=data['u'])
        if t != 0:
            sold = m.addVar(name='sold', obj=data['q'])
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=data['r'])
            m.addConstr(sold + unsatisfied == 0, uncertainty={'rhs':f2})
            m.addConstr(sold + recycled == buy_past)

    model_list.append(nvic)

@utils.timing
def solve_model_SDDP(m):
    m.discretize(random_state=1, n_samples=100)
    m_sddp = SDDP(m)
    m_sddp.solve(max_iterations=max_iter_solv, freq_evaluations=10, n_simulations=-1, tol=1e-3)
    return m_sddp.db[-1], m_sddp.first_stage_solution

# Solve models and collect the feasible ones for data set
feasible_models = []
for m in model_list:
    try:
        a = solve_model_SDDP(m)
        df_dict, aslema_df = m.write_cuts_to_df()
        print(aslema_df)
        feasible_models.append({'model': m, 'bound_solution_solvtime': a, 'cuts':df_dict})
    except Exception as e:
        print(f'Model {m} ist infeasible')
        print(e)

#print(feasible_models)

