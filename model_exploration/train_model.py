import torch

from transformers import get_linear_schedule_with_warmup
from torch import nn
import numpy as np
from collections import defaultdict


class TrainModel():

    def __init__(self, config_writer, model_name, optimizer, loss_fn, num_classes, num_epochs):
        self.log = config_writer
        self.model = model_name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_epochs = num_epochs

        
    def select_hardware(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log.print(f"DEVICE: {self.device}")

    def model_setup(self):
        self.model = self.model(self.num_classes)
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
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
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

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
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
                torch.save(self.model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc
        
        self.log.add("train_acc", history['train_acc'])
        self.log.add("train_loss", history['train_loss'])
        self.log.add("val_acc", history['val_acc'])
        self.log.add("val_loss", history['val_loss'])
        self.log.add("val_acc_max", max(history['val_acc']))
        return self.log

