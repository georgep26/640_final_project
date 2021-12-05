import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
import config.constants as constants


def check_imageid_exists(data_path, image_dir, image_id_col, out_path):
    df = pd.read_csv(data_path)
    mask = df[image_id_col].apply(lambda x: id_in_dir(x, image_dir))
    df = df[mask]
    df.to_csv(out_path)
    return df, sum(~mask)


def id_in_dir(id, image_dir):
    id = str(id)
    ids = os.listdir(image_dir)
    ids = list(map(lambda x: x.split(".")[0], ids))
    return id in ids


if __name__ == "__main__":
    id_in_dir("white", constants.data_dirs['images'])
    id_in_dir(1, constants.data_dirs['images'])

    for cfg in constants.clean_image_id:
        df, num_drop = check_imageid_exists(**cfg)
        print(f"DROPPED {num_drop} in preprocessing step")
