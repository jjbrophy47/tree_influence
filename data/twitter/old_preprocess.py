"""
Preprocess dataset.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def dataset_specific(random_state, test_size, n_instances):

    # retrieve dataset
    df = pd.read_csv('comments.csv', nrows=n_instances)

    # engineer some basic text features
    df['num_chs'] = df['text'].str.len()
    df['num_hsh'] = df['text'].str.count('#')
    df['num_men'] = df['text'].str.count('@')
    df['num_lnk'] = df['text'].str.count('http')
    df['num_rtw'] = df['text'].str.count('RT')
    df['num_uni'] = df['text'].str.count(r'(\\u\S\S\S\S)')

    # sequential features
    df['usr_msg_cnt'] = df.groupby('user_id').cumcount()

    # remove select columns
    remove_cols = ['com_id', 'user_id', 'text']
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)

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
    label = ['label']
    numeric = ['pagerank', 'triangle_count', 'core_id', 'out_degree', 'in_degree',
               'polarity', 'subjectivity', 'num_chs', 'num_hsh', 'num_men',
               'num_lnk', 'num_rtw', 'num_uni', 'usr_msg_cnt']
    categorical = list(set(columns) - set(numeric) - set(label))
    print('label', label)
    print('numeric', numeric)
    print('categorical', categorical)

    return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, n_instances=625000, n_bins=5):

    train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                                      test_size=test_size,
                                                                      n_instances=n_instances)

    # binarize inputs
    ct = ColumnTransformer([('kbd', KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense'), numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])
    train = ct.fit_transform(train_df)
    test = ct.transform(test_df)

    # binarize outputs
    le = LabelEncoder()
    train_label = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
    test_label = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

    # combine binarized data
    train = np.hstack([train, train_label]).astype(np.int32)
    test = np.hstack([test, test_label]).astype(np.int32)

    print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))
    print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    print('saving...')
    np.save('train.npy', train)
    np.save('test.npy', test)


if __name__ == '__main__':
    main()
