"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
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
    df = pd.read_csv('Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    df = df[~df['Data_Value'].isnull()]  # drop rows where the target is nan

    # split data into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = ['LocationDesc', 'Datasource', 'Topic', 'Data_Value_Unit', 'Data_Value_Type',
                   'Data_Value_Alt', 'Data_Value_Footnote', 'GeoLocation', 'ClassID', 'TopicID',
                   'QuestionID', 'DataValueTypeID', 'LocationID', 'StratificationCategory1',
                   'Stratification1', 'StratificationCategoryId1', 'StratificationID1',
                   'Data_Value_Footnote_Symbol']
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['Data_Value']
    features['numeric'] = ['YearStart', 'YearEnd', 'Low_Confidence_Limit', 'High_Confidence_Limit ',
                           'Sample_Size']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features,
                                                                logger=logger, objective='regression')

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, min., max.: {y_train.min()}, {y_train.max()}')
    logger.info(f'y_test.shape: {y_test.shape}, min., max.: {y_test.min()}, {y_test.max()}')

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
