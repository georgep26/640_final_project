import config.train_model_config as config_file
import config.constants as cst
import config.multimodal_config as config
from config.master_log import MasterLog
from model_exploration.train_multimodel import TrainModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from collections import defaultdict
from config.config_writer import ConfigWriter
from datetime import datetime
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models

if __name__ == "__main__":
    
    timestamp = datetime.now().strftime("%Y%d%m%H%M%S")
    model_name = config.bert_baseline_model['model_name']
    run_name = f"{model_name}_{timestamp}"
    output_dir = os.path.join(cst.output_dir, run_name)
    os.mkdir(output_dir)
    config_writer = ConfigWriter(output_dir)

    train_df = pd.read_csv(config.bert_baseline_data['train_data_loc'])
    test_df = pd.read_csv(config.bert_baseline_data['test_data_loc'])

    # TODO: Replace this with cross validation - added for testing purposes
    # train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=cst.RANDOM_SEED)

    # Implement k-fold
    kf = KFold(n_splits=4, random_state=None, shuffle=False)
    master_history = defaultdict(list)
    for train_index, val_index in kf.split(train_df):
        validation_df, train_df2 = train_df.iloc[val_index], train_df.iloc[train_index]
        # Get data loaders
        # train_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(train_df2, 
        #                                                                                 config.bert_baseline_data['text_col'],
        #                                                                                 config.bert_baseline_data['pred_col'],
        #                                                                                 config.bert_baseline_data['tokenizer'],
        #                                                                                 config.bert_baseline_data['max_len'],
        #                                                                                 config.bert_baseline_data['batch_size'],
        #                                                                                 config.bert_baseline_data['num_workers'])
        
        # validation_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(validation_df, 
        #                                                                                 config.bert_baseline_data['text_col'],
        #                                                                                 config.bert_baseline_data['pred_col'],
        #                                                                                 config.bert_baseline_data['tokenizer'],
        #                                                                                 config.bert_baseline_data['max_len'],
        #                                                                                 config.bert_baseline_data['batch_size'],
        #                                                                                 config.bert_baseline_data['num_workers'])

        # Reading image config 
        if cst.model_base["model"] == "resnet18":
            base_image_model = models.resnet18(pretrained=True)
        elif cst.model_base["model"] == "resnet50":
            base_image_model = models.resnet50(pretrained=True)
        elif cst.model_base["model"] == "resnet101":
            base_image_model = models.resnet50(pretrained=True)
        else:
            raise("invalid model selection")

        train_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(train_df2, 
                                                                                        cst.transform_config,
                                                                                        cst.dataset_config["train_image_dataset"],
                                                                                        config.bert_baseline_data['text_col'],
                                                                                        config.bert_baseline_data['pred_col'],
                                                                                        config.bert_baseline_data['tokenizer'],
                                                                                        config.bert_baseline_data['max_len'],
                                                                                        config.bert_baseline_data['batch_size'],
                                                                                        config.bert_baseline_data['num_workers'])
        
        validation_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(validation_df, 
                                                                                        cst.transform_config,
                                                                                        cst.dataset_config["train_image_dataset"],
                                                                                        config.bert_baseline_data['text_col'],
                                                                                        config.bert_baseline_data['pred_col'],
                                                                                        config.bert_baseline_data['tokenizer'],
                                                                                        config.bert_baseline_data['max_len'],
                                                                                        config.bert_baseline_data['batch_size'],
                                                                                        config.bert_baseline_data['num_workers'])

        session = TrainModel(config_writer, output_dir, base_image_model, cst.image_model_path, **config.bert_baseline_model)
        history = session.train(train_data_loader, validation_data_loader)


    test_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(test_df, 
                                                                                    cst.transform_config,
                                                                                    cst.dataset_config["train_image_dataset"],
                                                                                    config.bert_baseline_data['text_col'],
                                                                                    config.bert_baseline_data['pred_col'],
                                                                                    config.bert_baseline_data['tokenizer'],
                                                                                    config.bert_baseline_data['max_len'],
                                                                                    config.bert_baseline_data['batch_size'],
                                                                                    config.bert_baseline_data['num_workers'])
    
    test_acc = session.get_test_acc(test_data_loader)

    y_review_texts, y_pred, y_pred_probs, y_test = session.get_predictions(test_data_loader)

    config_writer.print(classification_report(y_test, y_pred))
    config_writer.print(confusion_matrix(y_test, y_pred))
    

    config_writer.write()
    final_val_acc = np.mean(config_writer.config['val_acc_max'])
    # Copy config file to output directory
    shutil.copy(config_file.__file__, output_dir)



    master_log = MasterLog()
    master_log.add_dict({"model_name": run_name, "model_path": output_dir, "validation_acc": final_val_acc, "test_acc": test_acc})
    master_log.write_log()
    