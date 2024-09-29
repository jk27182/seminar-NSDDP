"""Util module"""
import math
from datetime import datetime

from jax import numpy as jnp

from neural_sddp.piecewise_nn import cond_piecewise_nn as cpn


def get_date_time_string():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


# @jax.jit
def create_input_array(u, c, q, r, uncert):
    """Create input data in needed format"""
    A = jnp.array(
        [
            [1],
            [-1],
            [0],
            [0],
        ]
    )
    b = jnp.array(
        [
            [0],
            [-u],
            [0],
            [0],
        ]
    )
    W = jnp.array(
        [
            [1, 0],
            [1, 1],
            [1, 0],
            [0, 1],
        ]
    )
    Tec = jnp.array(
        [
            [0],
            [-1],
            [0],
            [0],
        ]
    )
    dim_mat = A.shape[0]
    c = c * jnp.ones((dim_mat,1))
    u = u * jnp.ones((dim_mat,1))
    r = r * jnp.ones((dim_mat,1))
    q = q * jnp.ones((dim_mat,1))
    uncert = uncert * jnp.ones((dim_mat,1))
    print("input before flat")
    print(jnp.column_stack((A, b, W, Tec,c, u, r, q, uncert)))
    print(jnp.column_stack((A, b, W, Tec,c, u, r, q, uncert)).shape)
    input_data = jnp.column_stack((A, b, W, Tec,c, u, r, q, uncert)).flatten("F")
    return input_data

data_coeff = {
    'c':23,
    'u': 10,
    'r':1,
    'q':4,
    'uncert':4,
}



def calc_loss(model, params, ftrs_test, trgts_test):
    loss = cpn.eval_loss(model, params, ftrs_test, 0, trgts_test)
    return loss


def get_training_data(features: jnp.array, targets: jnp.array, batchsize: int):
    n_batches = math.floor(len(features) / batchsize)
    # print("number of batches")
    # print(n_batches)
    for i in range(n_batches):
        upper_idx = (i + 1) * batchsize
        lower_idx = i * batchsize
        trgt = targets[lower_idx:upper_idx]
        ftr = features[lower_idx:upper_idx]
        yield ftr, trgt


