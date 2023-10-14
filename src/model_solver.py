import time

from msppy import MSLP
from msppy.solver import SDDP


def solve_msp(problem: MSLP, log_cons=0, log_file=0):
    tic = time.perf_counter()

    cuts_sddp = SDDP(problem)
    cuts_sddp.solve(
        freq_evaluations=1,
        n_simulations=-1,
        tol=10**(-4),
        max_time=0.06,
        logToConsole=log_cons,
        logFile=log_cons,
    )
    time_to_solve = time.perf_counter() - tic
    bounds = cuts_sddp.db
    policy_values = cuts_sddp.pv
    first_stage_solution = cuts_sddp.first_stage_solution
    stop_reason = cuts_sddp.stop_reason
    _, real_cuts = problem.write_cuts_to_df()

    return real_cuts, time_to_solve, bounds, policy_values, first_stage_solution, stop_reason


def construct_test_problem(features, cuts=None):
    u, c, q, r, uncertainty = features
    uncertainty = int(uncertainty)
    nvid = MSLP(T=2, sense=1)
    for t in range(2):
        m = nvid[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=c, ub=u)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=-q)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=-r)
            m.addConstr(sold + unsatisfied == 0,
            uncertainty={'rhs':range(uncertainty)})
            m.addConstr(sold + recycled == buy_past)
    if cuts is not None:
        cuts = cuts[0]
        # print(cuts)
        nvid.read_cuts_from_array(cuts)
    return nvid


def solve_probs_test_data(model, params, features_test, real_cut=None):
    solutions_data_wo_cuts = {}
    solutions_data_w_cuts = {}

    # feature = features_test
    stage = 0
    for idx, feature in enumerate(features_test):
        cuts = model.apply(params, feature, stage)

        problem_no_cuts = construct_test_problem(features=feature)
        problem_w_cuts = construct_test_problem(features=feature, cuts=cuts)

        real_cuts, time_to_solve, bounds, policy_values, first_stage_solution, stop_reason = solve_msp(problem_no_cuts)
        u, c, q, r, uncert = feature
        u = float(u)
        c = float(c)
        q = float(q)
        r = float(r)
        uncert = int(uncert)
        if "convergence" in stop_reason and "reached" in stop_reason:
            stop_reason = "convergence"
        solve_dict = {
            "time_to_solve": time_to_solve,
            "bounds":bounds,
            "policy_values" :policy_values,
            "first_stage_solution": first_stage_solution,
            "stop_reason": stop_reason
        }
        solutions_data_wo_cuts[u, c, q, r, uncert, idx] = solve_dict

        cut_predicted, time_to_solve_cut, bounds_cut, policy_values_cut, first_stage_solution_cut, stop_reason_cut = solve_msp(problem_w_cuts)

        if "convergence" in stop_reason_cut and "reached" in stop_reason_cut:
            stop_reason_cut = "convergence"

        solve_dict_cuts = {
            "time_to_solve" :time_to_solve_cut,
            "bounds" :bounds_cut,
            "policy_values" :policy_values_cut,
            "first_stage_solution" :first_stage_solution_cut,
            "stop_reason" :stop_reason_cut,
            "n_cuts": len(cuts[0]),
        }
        solutions_data_w_cuts[u, c, q, r, uncert, idx] = solve_dict_cuts

    return solutions_data_wo_cuts, solutions_data_w_cuts
