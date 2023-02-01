"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(test_size=0.2, seed=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # retrieve dataset
    start = time.time()
    feature_df = pd.read_csv(os.path.join('training_set_features.csv'))
    label_df = pd.read_csv(os.path.join('training_set_labels.csv'))
    data_df = feature_df.merge(label_df, on='respondent_id', how='left')
    logger.info('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # split data into train and test
    train_df, test_df = train_test_split(data_df,
                                         test_size=test_size,
                                         random_state=seed,
                                         stratify=data_df['seasonal_vaccine'])

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = ['respondent_id']
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['seasonal_vaccine']
    features['numeric'] = []
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features, logger=logger)

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, y_train.sum: {y_train.sum()}')
    logger.info(f'y_test.shape: {y_test.shape}, y_test.sum: {y_test.sum()}')

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
