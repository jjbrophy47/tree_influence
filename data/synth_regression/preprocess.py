"""
Generates a continuous attribute regression dataset.
"""
import os

import numpy as np
from sklearn.datasets import make_regression


def main(random_state=1,
         test_size=0.2,
         n_samples=10000,
         n_features=100,
         n_informative=20,
         effective_rank=2,
         tail_strength=0.6,
         noise=0.5):

    # retrieve dataset
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           effective_rank=effective_rank,
                           tail_strength=tail_strength,
                           noise=noise,
                           random_state=random_state)
    indices = np.arange(len(X))

    rng = np.random.default_rng(random_state)
    train_indices = rng.choice(indices, size=int(len(X) * (1 - test_size)), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    feature = ['x{}'.format(x) for x in range(X_train.shape[1])]

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    print('train:', X_train[:5], y_train[:5])
    print('test:', X_test[:5], y_test[:5])
    print('feature:', feature[:5])
    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
