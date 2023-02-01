"""
Preprocesses dataset but keep continuous variables.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def dataset_specific(random_state, test_size):

    # retrieve dataset
    df = pd.read_csv('Surgical-deepnet.csv')

    # remove nan rows
    nan_rows = df[df.isnull().any(axis=1)]
    print('nan rows: {}'.format(len(nan_rows)))
    df = df.dropna()

    # split into train and test
    indices = np.arange(len(df))
    n_train_samples = int(len(indices) * (1 - test_size))

    np.random.seed(random_state)
    train_indices = np.random.choice(indices, size=n_train_samples, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # categorize attributes
    columns = list(df.columns)
    label = ['complication']
    numeric = ['bmi', 'Age', 'ccsComplicationRate', 'ccsMort30Rate',
               'complication_rsi', 'hour', 'mortality_rsi']
    categorical = list(set(columns) - set(numeric) - set(label))

    return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, out_dir='continuous'):

    train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                                      test_size=test_size)

    # encode categorical inputs
    ct = ColumnTransformer([('kbd', 'passthrough', numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])
    train = ct.fit_transform(train_df)
    test = ct.transform(test_df)

    # binarize outputs
    le = LabelEncoder()
    train_label = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
    test_label = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

    # add labels
    train = np.hstack([train, train_label]).astype(np.float32)
    test = np.hstack([test, test_label]).astype(np.float32)

    print('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
    print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

    print('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
    print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    print('saving...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)


if __name__ == '__main__':
    main()
