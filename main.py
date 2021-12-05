print(sys.path)
import config.train_model_config as config
import model_exploration.train_model as train_session
import sys

if __name__ == "__main__":
   
    # Get data loaders
    train_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(config.bert_baseline_data['train_data_loc'], 
                                                                                    config.bert_baseline_data['text_col'],
                                                                                    config.bert_baseline_data['pred_col'],
                                                                                    config.bert_baseline_data['tokenizer'],
                                                                                    config.bert_baseline_data['max_len'],
                                                                                    config.bert_baseline_data['batch_size'],
                                                                                    config.bert_baseline_data['num_workers'])
    
    test_data_loader = config.bert_baseline_data['dataset_type'].create_data_loader(config.bert_baseline_data['test_data_loc'], 
                                                                                    config.bert_baseline_data['text_col'],
                                                                                    config.bert_baseline_data['pred_col'],
                                                                                    config.bert_baseline_data['tokenizer'],
                                                                                    config.bert_baseline_data['max_len'],
                                                                                    config.bert_baseline_data['batch_size'],
                                                                                    config.bert_baseline_data['num_workers'])

    train_session = train_session.TrainModel(**config.bert_baseline_model)

    train_session.train(train_data_loader)