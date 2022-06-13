import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

from data.preprocess import get_label_encoder_from_target, remove_extreme_values, split_data_into_features_and_targets


def main():
    df = pd.read_excel(Path('../../../data/Date_Fruit_Datasets.xlsx'), engine='openpyxl')
    df = remove_extreme_values(
        df=df,
        ignored_columns=['Class', 'ECCENTRICITY', 'KurtosisRB', 'KurtosisRG', 'EntropyRB', 'EntropyRG', 'EntropyRR',
                         'SOLIDITY', 'KurtosisRR', 'ROUNDNESS', 'SHAPEFACTOR_4', 'SHAPEFACTOR_2'],
        consider_all_columns=True
    )

    X, y = split_data_into_features_and_targets(df, target_name='Class')

    label_encoder = get_label_encoder_from_target(y=y)

    y = label_encoder.transform(y)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123,
                                                      stratify=y_train)

    clf = GradientBoostingClassifier()

    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)

    print(accuracy_score(y_val, y_pred_val))
    print(f1_score(y_val, y_pred_val, average='weighted'))


if __name__ == '__main__':
    main()
