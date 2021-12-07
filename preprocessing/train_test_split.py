import os.path
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.getcwd())
import config.constants as constants


def get_train_test(data_path, test_pct, seed):
    # TODO we may need to do something here with different frequencies of each label!

    # raw_df = pd.read_excel(data_path)
    raw_df = pd.read_csv(data_path)
    df_train, df_test = train_test_split(raw_df, test_size=test_pct, random_state=seed, shuffle=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    return df_train, df_test


def output_train_test(df, output_path):
    if not os.path.exists(output_path):
        print(f"WRITING {output_path}")
        df.to_csv(output_path, index=False)
    else:
        print(f"ALREADY FILE IN PATH {output_path}")


def check_file_exist(path):
    return os.path.exists(path)


if __name__ == "__main__":
    # check if train test dataset exists if not make it

    df_train, df_test = get_train_test(**constants.train_test_args['get_train_test'])
    output_train_test(df_train, **constants.train_test_args['output_dataset']['train'])
    output_train_test(df_test, **constants.train_test_args['output_dataset']['test'])



