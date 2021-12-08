# General imports 
import pandas as pd
import numpy as np
import os
import shutil
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



# Libraries for logging
from config.master_log import MasterLog
from config.config_writer import ConfigWriter

# Config files
import config.constants as cst
import config.multimodal_config as multi_config_file
import config.bert_config as bert_config_file

# Libraries for training
import model_exploration.train_bert as train_bert
import model_exploration.train_multimodel as train_mm


def run_BERT():
    '''
    Model setup for running bert
    '''

    timestamp = datetime.now().strftime("%Y%d%m%H%M%S")
    model_name = bert_config_file.bert_baseline_model['model_name']
    run_name = f"{model_name}_{timestamp}"
    output_dir = os.path.join(cst.output_dir, run_name)
    os.mkdir(output_dir)
    config_writer = ConfigWriter(output_dir)
    # Copy config file to output directory
    shutil.copy(bert_config_file.__file__, output_dir)

    train_df = pd.read_csv(bert_config_file.bert_baseline_data['train_data_loc'])
    test_df = pd.read_csv(bert_config_file.bert_baseline_data['test_data_loc'])

    # TODO: Replace this with cross validation - added for testing purposes
    # train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=cst.RANDOM_SEED)

    # Implement k-fold
    kf = KFold(n_splits=4, random_state=None, shuffle=False)
    master_history = defaultdict(list)
    for train_index, val_index in kf.split(train_df):
        validation_df, train_df2 = train_df.iloc[val_index], train_df.iloc[train_index]
        # Get data loaders
        train_data_loader = bert_config_file.bert_baseline_data['dataset_type'].create_data_loader(train_df2, 
                                                                                        bert_config_file.bert_baseline_data['text_col'],
                                                                                        bert_config_file.bert_baseline_data['pred_col'],
                                                                                        bert_config_file.bert_baseline_data['tokenizer'],
                                                                                        bert_config_file.bert_baseline_data['max_len'],
                                                                                        bert_config_file.bert_baseline_data['batch_size'],
                                                                                        bert_config_file.bert_baseline_data['num_workers'])
        
        validation_data_loader = bert_config_file.bert_baseline_data['dataset_type'].create_data_loader(validation_df, 
                                                                                        bert_config_file.bert_baseline_data['text_col'],
                                                                                        bert_config_file.bert_baseline_data['pred_col'],
                                                                                        bert_config_file.bert_baseline_data['tokenizer'],
                                                                                        bert_config_file.bert_baseline_data['max_len'],
                                                                                        bert_config_file.bert_baseline_data['batch_size'],
                                                                                        bert_config_file.bert_baseline_data['num_workers'])

        session = train_bert.TrainModel(config_writer, output_dir, **bert_config_file.bert_baseline_model)
        history = session.train(train_data_loader, validation_data_loader)


    test_data_loader = bert_config_file.bert_baseline_data['dataset_type'].create_data_loader(test_df, 
                                                                                    bert_config_file.bert_baseline_data['text_col'],
                                                                                    bert_config_file.bert_baseline_data['pred_col'],
                                                                                    bert_config_file.bert_baseline_data['tokenizer'],
                                                                                    bert_config_file.bert_baseline_data['max_len'],
                                                                                    bert_config_file.bert_baseline_data['batch_size'],
                                                                                    bert_config_file.bert_baseline_data['num_workers'])
    
    test_acc = session.get_test_acc(test_data_loader)

    y_review_texts, y_pred, y_pred_probs, y_test = session.get_predictions(test_data_loader)

    config_writer.print(classification_report(y_test, y_pred))
    # config_writer.print(confusion_matrix(y_test, y_pred))

    config_writer.write()
    final_val_acc = np.mean(config_writer.config['val_acc_max'])
    



    master_log = MasterLog()
    master_log.add_dict({"model_name": run_name, "model_path": output_dir, "validation_acc": final_val_acc, "test_acc": test_acc})
    master_log.write_log()

def run_multimodal():
    # Define logging inputs
    timestamp = datetime.now().strftime("%Y%d%m%H%M%S")
    model_name = multi_config_file.model_config['model_name']
    run_name = f"{model_name}_{timestamp}"
    output_dir = os.path.join(cst.output_dir, run_name)
    os.mkdir(output_dir)
    config_writer = ConfigWriter(output_dir)

    train_df = pd.read_csv(multi_config_file.data_config['train_data_loc'])
    test_df = pd.read_csv(multi_config_file.data_config['test_data_loc'])


    # Implement k-fold
    kf = KFold(n_splits=4, random_state=None, shuffle=False)
    master_history = defaultdict(list)
    for train_index, val_index in kf.split(train_df):
        validation_df, train_df2 = train_df.iloc[val_index], train_df.iloc[train_index]
       
        # Reading image config 
        if cst.model_base["model"] == "resnet18":
            base_image_model = models.resnet18(pretrained=True)
        elif cst.model_base["model"] == "resnet50":
            base_image_model = models.resnet50(pretrained=True)
        elif cst.model_base["model"] == "resnet101":
            base_image_model = models.resnet50(pretrained=True)
        else:
            raise("invalid model selection")

        train_data_loader = multi_config_file.data_config['dataset_type'].create_data_loader(train_df2, 
                                                                                        cst.transform_config,
                                                                                        cst.dataset_config["train_image_dataset"],
                                                                                        multi_config_file.data_config['text_col'],
                                                                                        multi_config_file.data_config['pred_col'],
                                                                                        multi_config_file.data_config['tokenizer'],
                                                                                        multi_config_file.data_config['max_len'],
                                                                                        multi_config_file.data_config['batch_size'],
                                                                                        multi_config_file.data_config['num_workers'])
        
        validation_data_loader = multi_config_file.data_config['dataset_type'].create_data_loader(validation_df, 
                                                                                        cst.transform_config,
                                                                                        cst.dataset_config["train_image_dataset"],
                                                                                        multi_config_file.data_config['text_col'],
                                                                                        multi_config_file.data_config['pred_col'],
                                                                                        multi_config_file.data_config['tokenizer'],
                                                                                        multi_config_file.data_config['max_len'],
                                                                                        multi_config_file.data_config['batch_size'],
                                                                                        multi_config_file.data_config['num_workers'])

        session = train_mm.TrainModel(config_writer, output_dir, base_image_model, cst.image_model_path, **multi_config_file.model_config)
        history = session.train(train_data_loader, validation_data_loader)


    test_data_loader = multi_config_file.data_config['dataset_type'].create_data_loader(test_df, 
                                                                                    cst.transform_config,
                                                                                    cst.dataset_config["train_image_dataset"],
                                                                                    multi_config_file.data_config['text_col'],
                                                                                    multi_config_file.data_config['pred_col'],
                                                                                    multi_config_file.data_config['tokenizer'],
                                                                                    multi_config_file.data_config['max_len'],
                                                                                    multi_config_file.data_config['batch_size'],
                                                                                    multi_config_file.data_config['num_workers'])
    
    test_acc = session.get_test_acc(test_data_loader)
    y_review_texts, y_pred, y_pred_probs, y_test = session.get_predictions(test_data_loader)

    # Record configuration and outputs
    config_writer.print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred)) # Currently causing an error when using config_writer    
    config_writer.write()
    final_val_acc = np.mean(config_writer.config['val_acc_max'])
    # Copy config file to output directory
    shutil.copy(multi_config_file.__file__, output_dir)

    # Update master log of all model runs
    master_log = MasterLog()
    master_log.add_dict({"model_name": run_name, "model_path": output_dir, "validation_acc": final_val_acc, "test_acc": test_acc})
    master_log.write_log()


if __name__ == "__main__":
    # run_BERT()
    run_multimodal()
    
    