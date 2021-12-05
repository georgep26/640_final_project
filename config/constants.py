import os


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
data_dirs = {
    "raw_data": os.path.join(data_root, "raw_data"),
    "data_root": data_root,
    "images": os.path.join(data_root, "images")
}

data_paths = {
    "raw_text_data": "data/data.xlsx",
    "raw_text_data_csv": "data/data.csv",
    "train_data": os.path.join(data_dirs['raw_data'], "data_train.csv"),
    "test_data": os.path.join(data_dirs['raw_data'], "data_test.csv")
}

train_test_args = {
  "get_train_test": {
    "seed": 0,
    "data_path": data_paths['raw_text_data'],
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

dataset_config = {
    "train_image_dataset": {
        # "data_path": data_paths['train_data'],
        "data_path": data_paths['raw_text_data_csv'],
        "image_dir": data_dirs['images'],
        "label_col": "Q3 Theme1",
        "image_id_col": "imageID",
        "train_data": True
    }
}