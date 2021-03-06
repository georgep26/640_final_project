import os

RANDOM_SEED = 42

q1_relation_lables = {
    1: "Yes",
    2: "No"
}

q2_focus_labels = {
    1: "The story focuses on one incident or event related to gun violence",
    2: "The story focuses on the issue of gun violence as an ongoing problem"
}

q3_theme_labels = {
    1: "Gun/2nd Amendment Rights",
    2: "Gun Control/Regulation",
    3: "Politics",
    4: "Mental Health",
    5: "School or Public Space Safety",
    6: "Race/Ethnicity",
    7: "Public Opinion",
    8: "Society/Culture",
    9: "Economic Consequences",
    99: "None of the above (No Label)",
}

data_root = "data"
modeling_root = "model_exploration"
data_dirs = {
    "raw_data": os.path.join(data_root, "raw_data"),
    "data_root": data_root,
    "images": os.path.join(data_root, "images"),
    "preprocessed": os.path.join(data_root, "preprocessed_data"),
    "model_results": os.path.join(modeling_root, "model_results")
}

data_paths = {
    "raw_text_data": "data/data.xlsx",
    "raw_text_data_csv": "data/data.csv",
    "train_data": os.path.join(data_dirs['raw_data'], "data_rebal_train.csv"),
    "test_data": os.path.join(data_dirs['raw_data'], "data_rebal_test.csv"),
    "preprocessed_train_data": os.path.join(data_dirs['preprocessed'], "data_rebal_train.csv"),
    "preprocessed_test_data": os.path.join(data_dirs['preprocessed'], "data_rebal_test.csv"),
    "raw_rebalanced_data": os.path.join(data_dirs['preprocessed'], "rebalanced_data.csv")
}

output_dir = "model_exploration/model_results"

master_log_path = "model_exploration/model_results/"

##################################################
# config for train test split
##################################################

train_test_args = {
  "get_train_test": {
    "seed": 0,
    "data_path": data_paths['raw_rebalanced_data'],
    "test_pct": 0.20
  },

  "output_dataset": {
      "train": {
          "output_path": data_paths['train_data']
      },
      "test":{
          "output_path": data_paths['test_data']
      }
  }
}

##################################################
##################################################

##################################################
# config for id cleaning
##################################################

clean_image_id = [
    {
        "data_path": data_paths['train_data'],
        "image_dir": data_dirs['images'],
        "image_id_col": "imageID",
        "out_path": data_paths['preprocessed_train_data']
    },
    {
        "data_path": data_paths['test_data'],
        "image_dir": data_dirs['images'],
        "image_id_col": "imageID",
        "out_path": data_paths['preprocessed_test_data']
    }
]


##################################################
##################################################

##################################################
# config for dataset rebalance
##################################################

rebalance_config = {
    "rebal_input": {
        "feature_col": "Q3 Theme1",
        "unique_id_col": "imageID"
    },
    "data_path": data_paths['raw_text_data'],
    "out_path": data_paths['raw_rebalanced_data']
}


##################################################
##################################################

##################################################
# Config for unimodal image learning
##################################################

dataset_config = {
    "train_image_dataset": {
        "image_dir": data_dirs['images'],
        "label_col": "Q3 Theme1",
        "image_id_col": "imageID",
    },
    "val_image_dataset": {
        "image_dir": data_dirs['images'],
        "label_col": "Q3 Theme1",
        "image_id_col": "imageID",
    },
    "train_downsample_frac": .001
}

model_base = {
    "model": "resnet101"
}

model_config = {
    "num_labels": 9,
    "dropout": 0.5
}

loader_config = {
    "batch_size": 16
}

transform_config = {
    "train": {
        "image_shape": (224, 224),
        "horizontal_flip": {
            "p": 0.5
        },
        "rotation": {
            "degrees": 25
        }
        },
    "inference": {
        "image_shape": (224, 224),
    }
}

train_config = {
    "num_epochs": 10,
    "learning_rate": 2e-5,
    "train_ds_config": dataset_config['train_image_dataset'],
    "val_ds_config": dataset_config['val_image_dataset'],
    "model_config": model_config,
    "loader_config": loader_config
}

description = "unimodal resnet101 transfer learned slightly smaller batch size higher transform params higher epoch"

image_model_path = "model_exploration/model_results/unimodal_image_20210512215552/best_model_state.bin"

##################################################
##################################################
