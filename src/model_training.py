import pickle

import utils
from neural_sddp.piecewise_nn import cond_piecewise_nn as cpn
from utils import get_training_data


def get_model_params(
    model,
    key,
    optimizer,
    stage,
    features,
    targets,
    n_pieces,
    n_layers,
    hidden_size,
    n_epochs,
    tolerance,
):
    try:
        with open(f"params_trained_model_{n_pieces}_{hidden_size}_{n_layers}_{n_epochs}_{tolerance}", "rb") as f:
            params = pickle.load(f)

    except FileNotFoundError:
        params = model.init(key, features, stage)
        params, loss = train_model(
        model, optimizer, params, n_epochs, tolerance, features, targets, stage=0,
    )
        with open(f"params_trained_model_{n_pieces}_{hidden_size}_{n_layers}_{n_epochs}_{tolerance}", "wb") as f:
            pickle.dump(params, f)

    return params


def train_model(model, optimizer, params, n_epochs, tolerance, features, targets, stage):
    opt_state = optimizer.init(params)

    print("Start Training")
    for epoch in range(n_epochs):
        for feat, targ in get_training_data(features, targets, batchsize=32):
            params, opt_state, loss = cpn.train_step(model, feat, stage, targ, params, optimizer, opt_state)

        print(f'Epoch {epoch}, loss: {loss}')
        if loss < tolerance:
            print(f'The Loss {loss} lower than tolerance: {tolerance} in epoch {epoch}')
            break
    return params, loss
