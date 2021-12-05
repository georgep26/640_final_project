import torch

from transformers import get_linear_schedule_with_warmup
from torch import nn
import numpy as np
from collections import defaultdict


class TrainModel():

    def __init__(self, model_name, optimizer, loss_fn, num_classes, num_epochs):
        self.model = model_name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_epochs = num_epochs

        
    def select_hardware(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {self.device}")

    def model_setup(self):
        self.model = self.model(10)
        self.model.to(self.device)
    
    def train_epoch(self, data_loader):
        self.model = self.model.train()

        losses = []
        correct_predictions = 0
        n_examples = len(data_loader)
        
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
        n_examples = len(data_loader)

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


    def train(self, data_loader):
        print("Training model...")
        self.select_hardware()
        self.model_setup()

        self.optimizer = self.optimizer(self.model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(data_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        self.loss_fn.to(self.device)
        
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.num_epochs):

            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            train_acc, train_loss = self.train_epoch(data_loader)
            
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(data_loader)
            
            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc
