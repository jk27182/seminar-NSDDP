"""Portfolio optimization problem, verwenden von newsvendor problem als Benchmark"""
import sys
from collections import defaultdict

import numpy as np
from msppy.msp import MSLP
from msppy.msp import MSLP
from msppy.solver import SDDP
from msppy.evaluation import Evaluation, EvaluationTrue
from msppy.utils.examples import construct_nvm
# from msppy.solver import SDDP, Extensive
# from msppy.evaluation import Evaluation, EvaluationTrue


T = 2
nvm = MSLP(T=T, sense=1, bound=500)
max_iter = 20
lp_path = sys.argv[1]

def sample_path_generator(random_state, size):
    a = np.zeros([size, T, 1])
    for t in range(1, T):
        a[:, t, :] = (0.5 * a[:, t - 1, :]
                      + random_state.lognormal(2.5, 1, size=[size, 1]))
    return a

nvm.add_Markovian_uncertainty(sample_path_generator)

state_var_name = 'bought'
for t in range(T):
    m = nvm[t]
    buy_now, buy_past = m.addStateVar(name=state_var_name, obj=1.0)
    if t != 0:
        sold = m.addVar(name='sold', obj=-2)
        unsatisfied = m.addVar(name='unsatisfied')
        recycled = m.addVar(name='recycled', obj=-0.5)
        m.addConstr(sold + unsatisfied == 0, uncertainty_dependent={'rhs': 0})
        m.addConstr(sold + recycled == buy_past)

# nvm = construct_nvm()
nvm.discretize(n_Markov_states=10, n_sample_paths=1000, method='SA')
# nvm_ext = Extensive(nvm)
# nvm_ext.solve(outputFlag=0)
# nvm_ext.first_stage_solution
nvm_sddp = SDDP(nvm)
nvm_sddp.solve(max_iterations=max_iter, logToConsole=1)
print(nvm_sddp.cut_type_list)
print(nvm_sddp.cut_T)
print(nvm_sddp.db[-1])
print(nvm_sddp.first_stage_solution)

def get_cut_attr(s_model, state_var_name='bought', theta='alpha', sense=-1):
    '''s_model ist das Grundmodell, nicht nach Anwendung von SDDP'''

    nested_def_dict = lambda: defaultdict(tuple)
    nested_def_dict_outer = lambda: defaultdict(nested_def_dict)
    cut_list = defaultdict(nested_def_dict_outer)

    for t in range(s_model.T):
        m_t = s_model[t]
        for k, m_i in enumerate(m_t):
            value_func = m_i.getVarByName(theta)
            x_var = m_i.getVarByName(state_var_name)
            # if stage t has a value function, so every stage except the last one
            if value_func:
                for i_cnstr, cnstr in enumerate(m_i.getConstrs()):
                    value_func_coeff = m_i.getCoeff(cnstr, value_func)
                    x_var_coeff = m_i.getCoeff(cnstr, x_var)
                    rhs = cnstr.RHS
                    cut_list[t][k][i_cnstr] = (sense * x_var_coeff, rhs, cnstr)
    return cut_list


tmp = get_cut_attr(nvm)
print(tmp[0][0][0])
# horrible for loops
cut_list = []
for t in range(nvm.T):
    m_t = nvm.models[t]
    for k, m_i in enumerate(m_t):
        # m_i.write('aslema.mps')
        alpha_var = m_i.getVarByName('alpha')
        state_var = m_i.getVarByName(state_var_name)
        if alpha_var:
            for cnstr in m_i.getConstrs():
                # print("Constraint %s: sense %s, RHS=%f" % (cnstr.ConstrName, cnstr.Sense, cnstr.RHS))
                row = m_i.getRow(cnstr)
                # print('alpha coeff', m_i.getCoeff(cnstr, alpha_var))
                # print('state+var coeff', m_i.getCoeff(cnstr, state_var))
                # print(row, cnstr.RHS)
                cut_list.append([row, cnstr.RHS])
                for k in range(row.size()):
                    if row.getVar(k).VarName == 'bought' or row.getVar(k).VarName == 'alpha':
                        print("Variable %s, coefficient %f" % (row.getVar(k).VarName, row.getCoeff(k)))
                        nvm.write(path=lp_path, suffix='.lp')
# # print('='*50)
# # print(cut_list)
