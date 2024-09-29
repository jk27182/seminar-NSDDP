import pickle

from jax import numpy as jnp
from sklearn.model_selection import train_test_split


def get_train_test_data(n_pieces, n_layers, hidden_size, test_size, data):
    try:
        with open(f'preprocessed_data_{n_pieces}_{hidden_size}_{n_layers}', 'rb') as f:
            features, targets = pickle.load(f)

    except FileNotFoundError:
        features, targets = data_preprocess(data, n_pieces)
        with open(f'preprocessed_data_{n_pieces}_{hidden_size}_{n_layers}', 'wb') as f:
            pickle.dump((features, targets), f)

    feat_train, feat_test, trgts_train, trgts_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=1,
    )

    return features, targets, feat_train, feat_test, trgts_train, trgts_test


def data_preprocess(data, n_pieces):
    feature = []
    target = []
    for feat, cuts in data.items():
    # Do not use last column since that is just the stage number
        if len(cuts) >= n_pieces:
            feature.append(feat)
            target.append(cuts.iloc[:n_pieces,:].values)
    features = jnp.array(feature)
    targets = jnp.array(target)

    return features, targets
