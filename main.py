import config.constants as cst
import config.train_model_config as config
import model_exploration.train_model as train_session
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":
    
    train_df = pd.read_csv(config.bert_baseline_data['train_data_loc'])
    test_df = pd.read_csv(config.bert_baseline_data['test_data_loc'])

    # TODO: Replace this with cross validation - added for testing purposes
    train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=cst.RANDOM_SEED)
    
    # Get data loaders
    train_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(train_df, 
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


    test_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(test_df, 
                                                                                    config.bert_baseline_data['text_col'],
                                                                                    config.bert_baseline_data['pred_col'],
                                                                                    config.bert_baseline_data['tokenizer'],
                                                                                    config.bert_baseline_data['max_len'],
                                                                                    config.bert_baseline_data['batch_size'],
                                                                                    config.bert_baseline_data['num_workers'])

    train_session = train_session.TrainModel(**config.bert_baseline_model)

    train_session.train(train_data_loader, validation_data_loader)
    