from typing import List, TypeVar

import numpy as np
from matplotlib import pyplot as plt

SolTimeCuts = TypeVar("SolTimeCuts")

def plot_comparison_bar(sol_time_no_cut, sol_time_cut: List[SolTimeCuts], barwidth=0.25, scale=0.8, file_name=None):
    n_sol_instances = len(sol_time_no_cut)
    print("number of sol instances")
    print(n_sol_instances)
    pos1 = np.arange(n_sol_instances)*1
    plt.bar(pos1, sol_time_no_cut, width=barwidth, label="SDDP")

    for counter, prob_instance in enumerate(sol_time_cut):
        pos_instance = pos1 + barwidth*(counter + 1)*scale
        sol_time_instance = prob_instance[0]
        n_cuts = prob_instance[1]
        print(prob_instance)
        plt.bar(pos_instance, sol_time_instance, width=barwidth, label=f"NSDDP-{n_cuts}", align="center")

    plt.xlabel("Problem instance")
    if len(sol_time_cut) % 2 == 0:
        plt.xticks(
            pos1 + scale*barwidth*(len(sol_time_cut) - 1),
            [i + 1 for i in range(n_sol_instances)],
        )
    else:
        plt.xticks(
            pos1 + scale*barwidth/2 * len(sol_time_cut),
            [i + 1 for i in range(n_sol_instances)],
        )
    plt.ylabel("Computing time in sec.")
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 1.05), ncol=1, framealpha=1)
    plt.show()
    if file_name:
        plt.savefig(f"{file_name}.png")


def compare_solution_times(data_w_cuts, data_no_cuts):
    total_instances = len(data_no_cuts)
    sol_time_no_cut = []
    sol_time_cut = []
    solved_instances = 0
    for prob_instance in data_no_cuts:
        no_cut_vals = data_no_cuts[prob_instance]
        cut_vals = data_w_cuts[prob_instance]
        if cut_vals["stop_reason"] == "convergence":
            sol_time_no_cut.append(no_cut_vals["time_to_solve"])
            sol_time_cut.append(cut_vals["time_to_solve"])
            solved_instances += 1
            n_cuts = cut_vals["n_cuts"]

    return total_instances, solved_instances, sol_time_no_cut, (sol_time_cut, n_cuts)


def plot_loss_per_epoch(loss_test: List, loss_train: List, epochs: List, file_name=None):
    import matplotlib.pyplot as plt
    plt.plot(epochs, loss_train, label="Train loss")
    plt.plot(epochs, loss_test, label="Test loss")
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    if file_name:
        return plt.savefig(file_name)
    plt.show()
