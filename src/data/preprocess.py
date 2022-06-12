import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List
import numpy as np
from scipy import stats
from loguru import logger


def split_data_into_features_and_targets(df: pd.DataFrame, target_name='Class') -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = df.drop([target_name], axis=1)
    target = df[target_name]
    return features, target


def get_label_encoder_from_target(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


def remove_extreme_values(
        df: pd.DataFrame,
        columns: List[str] = None,
        ignored_columns: List[str] = None,
        consider_all_columns: bool = False
) -> pd.DataFrame:
    # FIXME: Add documentation

    if consider_all_columns:
        columns = list(df.columns)

    if ignored_columns is None:
        ignored_columns = []

    considered_columns = set(columns) - set(ignored_columns)

    total_rows_deleted = 0

    for column in considered_columns:
        n_rows_before = len(df)
        df = df[(np.abs(stats.zscore(df[[column]])) < 3).all(axis=1)]
        n_rows_after = len(df)
        rows_deleted = n_rows_before - n_rows_after
        logger.info(f'Removed {rows_deleted} extreme values from {column}.')
        total_rows_deleted += rows_deleted

    logger.info(f'In total removed {total_rows_deleted} rows.')

    return df


if __name__ == '__main__':
    pass
