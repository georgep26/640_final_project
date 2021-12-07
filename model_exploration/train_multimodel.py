import torch
import os
from transformers import get_linear_schedule_with_warmup
from torch import nn
import numpy as np
from collections import defaultdict
import torch.nn.functional as F


class TrainModel():

    def __init__(self, config_writer, output_dir, base_image_model, image_model_path, model_name, model_class, optimizer, loss_fn, num_classes, num_epochs, dropout, bert_model_path):
        self.log = config_writer
        self.output_dir = output_dir
        self.model_name = model_name
        self.model = model_class
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.base_image_model = base_image_model
        self.image_model_path = image_model_path
        self.bert_model_path = bert_model_path

        
    def select_hardware(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log.print(f"DEVICE: {self.device}")

    def model_setup(self):
        self.model = self.model(self.num_classes, self.base_image_model, self.image_model_path, self.bert_model_path, self.dropout)
        self.model.to(self.device)
    
    def train_epoch(self, data_loader):
        self.model = self.model.train()

        losses = []
        correct_predictions = 0
        # The number of examples is the total record count (so the size of the dataset)
        n_examples = len(data_loader.dataset)
        
        
        for d in data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            sentiment = d["classification"].to(self.device)
            image = d["image"].to(self.device)

            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image = image
            )

            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, sentiment)

            correct_predictions += torch.sum(preds == sentiment)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(self, data_loader):
        self.model = self.model.eval()

        losses = []
        correct_predictions = 0
        # TODO: Might need to pass in the number of examples from the dataframe - dataset should have same length as df
        n_examples = len(data_loader.dataset)
        

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                sentiment = d["classification"].to(self.device)
                image = d["image"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image = image
                )
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, sentiment)

                correct_predictions += torch.sum(preds == sentiment)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)


    def train(self, train_data_loader, validation_data_loader):
        self.log.print("Training model...")
        self.select_hardware()
        self.model_setup()

        self.optimizer = self.optimizer(self.model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_data_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        self.loss_fn.to(self.device)
        
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.num_epochs):

            self.log.print(f'Epoch {epoch + 1}/{self.num_epochs}')
            self.log.print('-' * 10)

            train_acc, train_loss = self.train_epoch(train_data_loader)
            
            self.log.print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(validation_data_loader)
            
            self.log.print(f'Val   loss {val_loss} accuracy {val_acc}')
            self.log.print("")

            history['train_acc'].append(train_acc.item())
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc.item())
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model_state.bin'))
                best_accuracy = val_acc
        
        self.log.add("train_acc", history['train_acc'])
        self.log.add("train_loss", history['train_loss'])
        self.log.add("val_acc", history['val_acc'])
        self.log.add("val_loss", history['val_loss'])
        self.log.add("val_acc_max", max(history['val_acc']))
        self.log.print(str(self.model))
        return self.log

    def get_test_acc(self, test_data_loader):
        test_acc, _ = self.eval_model(test_data_loader)
        return test_acc.item()
    
    def get_predictions(self, test_data_loader):
        model = self.model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in test_data_loader:
                texts = d["text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["classification"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values

