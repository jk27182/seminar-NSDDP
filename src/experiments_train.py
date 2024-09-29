import pickle
import itertools

import jax
import numpy as np
import optax

import model_solver
import neural_sddp.piecewise_nn.cond_piecewise_nn as cpn
import data_preprocessing as pp
import model_training as mdlt
import postprocessing as pop

data_n_pieces = [1, 2, 3]
# 3 ist das maximum was an Pieces klappt.
# Bei 4 (6 parameter zu prediction) verschiedenen Werten eig aber nicht logisch,
# eher 2 (4 parameter predicten)verwenden
data_n_layers = [1]
data_hidden_size = [128]

# Anzahl von Variablen bleibt fix
N_VARS = 1
# Mit wie vielen Problemen soll Solution time verglichen werden
N_INSTANCES_TO_COMPARE_START = 60
N_INSTANCES_TO_COMPARE_END = 65

SEED = 1
N_EPOCHS = 30
TOLERANCE = 1e-6
TEST_SIZE = 0.3
LEARNING_RATE = 0.0004
OPTIMIZER = optax.adam(learning_rate=LEARNING_RATE)

KEY = jax.random.PRNGKey(SEED)

path = "datauniform04_06_2022 01:20:48"
with open(path, "rb") as f:
    data = pickle.load(f)
    # Es verändern sich 4 Werte jeweils

solution_time_with_cuts = []
for n_pieces, n_layers, hidden_size in itertools.product(data_n_pieces, data_n_layers, data_hidden_size):
    model = cpn.CondPiecewiseNN(
        num_vars=N_VARS,
        num_pieces=n_pieces,
        num_layers=n_layers,
        num_stages=2,
        hidden_size=hidden_size,
    )
    stage = 0
    features, targets, feat_train, feat_test, trgts_train, trgts_test = pp.get_train_test_data(
        n_pieces,
        n_layers,
        hidden_size,
        TEST_SIZE,
        data,
    )
    print("Menge an Daten zum training")
    print(len(features))
    print("Menge Trainings Daten")
    print(len(feat_train))
    print("Menge Test Daten")
    print(len(feat_test))

    params = mdlt.get_model_params(
        model,
        KEY,
        OPTIMIZER,
        stage,
        feat_train,
        trgts_train,
        n_pieces,
        n_layers,
        hidden_size,
        N_EPOCHS,
        TOLERANCE,
    )
    samples = slice(N_INSTANCES_TO_COMPARE_START, N_INSTANCES_TO_COMPARE_END)
    feature_data_point = feat_test[samples]
    real_data_point = trgts_test[samples]
    print("Anwendung der cuts")
    solutions_data_wo_cuts, solutions_data_w_cuts = model_solver.solve_probs_test_data(model, params, feature_data_point, real_data_point)

    total_instances, solved_instances, sol_time_no_cut, sol_time_cut = pop.compare_solution_times(
        solutions_data_w_cuts,
        solutions_data_wo_cuts,
    )
    print("total_instances")
    print(total_instances)
    print('Solved instances')
    print(solved_instances)
    print(f"Percentage of solved instances: {(solved_instances / total_instances)*100}%")

    t_mean_no_cuts = np.mean(sol_time_no_cut)
    t_mean_cuts = np.mean(sol_time_cut[0])
    print("Difference between solution with cuts and solution with no cuts")
    print(f"{(1 - t_mean_cuts / t_mean_no_cuts)*100}%")

    solution_time_with_cuts.append(sol_time_cut)

pop.plot_comparison_bar(sol_time_no_cut, solution_time_with_cuts, file_name="computing_time_128hidden_neurons")
