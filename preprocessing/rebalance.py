import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
import config.constants as constants
from sklearn.utils import resample


def rebalance_dataset(df, feature_col, unique_id_col):
    balanced_df = pd.DataFrame()

    max_count = df.groupby(feature_col).count()[unique_id_col].max()
    max_label = df.index[df.groupby(feature_col).count()[unique_id_col].idxmax()]

    for group, sub_df in df.groupby(feature_col):
        replace = False if group == max_label else True
        # print(f"Group: {group}, Replace: {replace}")
        temp_df = resample(sub_df, replace=replace, n_samples=max_count)
        balanced_df = pd.concat([balanced_df, temp_df])

    return balanced_df


if __name__ == "__main__":

    df = pd.read_excel(constants.rebalance_config['data_path'])
    rebal_df = rebalance_dataset(df, **constants.rebalance_config['rebal_input'])
    rebal_df.to_csv(constants.rebalance_config['out_path'])

