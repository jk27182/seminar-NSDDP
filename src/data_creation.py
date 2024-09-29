from collections import Counter
import pickle
from typing import Dict

from msppy.msp import MSLP
from msppy.solver import SDDP
import numpy as np

import utils



# Sense -1 ist maximimize und 1 ist minize. Default ist minimize
def construct_problem(T, u, c, q, r, uncertainty):
    """Construction of problem.

    Args:
        u: upper bound for stage variable
        c: price per newspaper
        q: selling price per newspaper
        r: return/recycling price, r < c
        uncertainty (_type_): wie viele scenarios soll es bei der Gleichverteilung geben.

    Returns:
        _type_: Das Problem. Ist eine Liste von Problemen
    """
    nvid = MSLP(T=T)
    for t in range(2):
        m = nvid[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=c, ub=u)
        else:
            _, buy_past = m.addStateVar(name='bought', ub=u)
            sold = m.addVar(name='sold', obj=-q)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=-r)
            m.addConstr(sold + unsatisfied == 0,
            uncertainty={'rhs':range(uncertainty)})
            m.addConstr(sold + recycled == buy_past)

    return nvid

    
def find_most_frequent_cut(data: Dict):
    value_list = list(map(len, data.values()))
    occur_dict = Counter(value_list)
    len_most_frquent_cut = max(occur_dict, key=occur_dict.get)
    return len_most_frquent_cut


def trim_data_to_most_freq_cuts(data: Dict, len_override=None):
    if len_override:
        len_most_frq_cut = len_override
    else:
        len_most_frq_cut = find_most_frequent_cut(data)
    trimmed_data = {}
    for k, v in data.items():
        if len(v) == len_most_frq_cut:
            trimmed_data[k] = v
    return trimmed_data, len_most_frq_cut

versuch = "uniform"
data = {}
samples = 10_000
prct = 0.4
uncertainty = 11
np.random.seed(1)
u = np.random.randint(25 - prct*25, 25 + prct*25, (samples,))
c = np.random.normal(1, 0.4, (samples,))
q = np.random.normal(2, prct*2, (samples,))
r = np.random.normal(0.5, prct*0.5, (samples,))
# uncertainty = np.random.normal(10, 0, (samples,))
if __name__ == "__main__":
    for sample in range(samples):
        try:
            T_iter = 2
            u_iter = u[sample]
            c_iter = c[sample]
            q_iter = q[sample]
            r_iter = r[sample]
            # uncertainty_iter = uncertainty

            prob = construct_problem(T=T_iter, u=u_iter, c=c_iter, q=q_iter, r=r_iter, uncertainty=uncertainty)
            nvm_sddp = SDDP(prob)
            nvm_sddp.solve(freq_evaluations=1, n_simulations=-1, tol=10**(-4), logToConsole=0)
            df_dict, df_cuts = prob.write_cuts_to_df()
            data[u_iter, c_iter, q_iter, r_iter, uncertainty] = df_cuts
            # print('nvm_sddp.db[-1]',nvm_sddp.db[-1])
            # print('nvm_sddp.first_stage_solution',nvm_sddp.first_stage_solution)
        except Exception as e:
            print("Iter number", sample)
            print("Problem ist nicht loesbar")
            print(e)
        # Das problem beschreibt eigentlich ein maximierungs problem, Wertfunktion muss noch mit minus 1 mulitpliziert werden.
        # Minus alphha beschreibt dann den erwarteten Profit on Sales und returns (siehe Birge Louveaux)

    # trimmed_data, len_most_frq_cut = trim_data_to_most_freq_cuts(data, len_override=2)
    # print(len_most_frq_cut)
    # with open(f"data_trimmed_to_{2}.pckl", "wb") as f:
    #     pickle.dump(trimmed_data, f)
    t_stamp = utils.get_date_time_string()
    t_stamp = t_stamp.replace('/', '_')
    with open("data" + versuch + t_stamp, "wb") as f:
        pickle.dump(data, f)

    # trimmed_data, len_most_frq_cut = trim_data_to_most_freq_cuts(data)
    # with open("data_cut_len" + versuch, "wb") as f:
    #     pickle.dump(len_most_frq_cut, f)

    # with open(f"data_trimmed_to_{len_most_frq_cut}" + versuch, "wb") as f:
    #     pickle.dump(trimmed_data, f)
