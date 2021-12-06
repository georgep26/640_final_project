import config.constants as cst
import config.train_model_config as config
from model_exploration.train_model import TrainModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from collections import defaultdict
from config.config_writer import ConfigWriter
from datetime import datetime
import os

if __name__ == "__main__":
    
    file_name = datetime.now().strftime("%Y%d%m%H%M%S")
    output_dir = os.path.join(cst.output_dir, f"bert_model_{file_name}")
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
        train_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(train_df2, 
                                                                                        config.bert_baseline_data['text_col'],
                                                                                        config.bert_baseline_data['pred_col'],
                                                                                        config.bert_baseline_data['tokenizer'],
                                                                                        config.bert_baseline_data['max_len'],
                                                                                        config.bert_baseline_data['batch_size'],
                                                                                        config.bert_baseline_data['num_workers'])
        
        validation_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(validation_df, 
                                                                                        config.bert_baseline_data['text_col'],
                                                                                        config.bert_baseline_data['pred_col'],
                                                                                        config.bert_baseline_data['tokenizer'],
                                                                                        config.bert_baseline_data['max_len'],
                                                                                        config.bert_baseline_data['batch_size'],
                                                                                        config.bert_baseline_data['num_workers'])

        session = TrainModel(config_writer, **config.bert_baseline_model)
        history = session.train(train_data_loader, validation_data_loader)
        # master_history['train_acc'].append(history['train_acc'])
        # master_history['train_loss'].append(history['train_loss'])
        # master_history['val_acc'].append(history['val_acc'])
        # master_history['val_loss'].append(history['val_loss'])


    
    # config_writer.add("model_history", master_history)
    config_writer.write()


    test_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(test_df, 
                                                                                    config.bert_baseline_data['text_col'],
                                                                                    config.bert_baseline_data['pred_col'],
                                                                                    config.bert_baseline_data['tokenizer'],
                                                                                    config.bert_baseline_data['max_len'],
                                                                                    config.bert_baseline_data['batch_size'],
                                                                                    config.bert_baseline_data['num_workers'])
    