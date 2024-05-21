import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import time

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sst2_train = load_dataset("sst2", split="train")
sst2_val = load_dataset("sst2", split="validation")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [d["label"] for d in df]
        self.texts = [
            tokenizer(
                d["sentence"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for d in df
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, num_classes, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

    return model


def evaluate(model, data):

    dataloader = torch.utils.data.DataLoader(Dataset(data), batch_size=8)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    correct_predictions = 0
    total_time = 0

    with torch.no_grad():

        for input_data, labels in dataloader:

            labels = labels.to(device)
            mask = input_data["attention_mask"].to(device)
            input_ids = input_data["input_ids"].squeeze(1).to(device)

            start_time = time.time()

            outputs = model(input_ids, mask)

            end_time = time.time()
            total_time += end_time - start_time

            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = correct_predictions / len(data)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Evaluation Time per Sample: {total_time / len(data)} seconds")

    return accuracy


EPOCHS = 3
#model = BertClassifier(2)
LR = 2e-5

# Train model
#trained_model = train(model, sst2_train, sst2_val, LR, EPOCHS)

# Evaluate model
#evaluate(trained_model, sst2_val)

# Save model
#torch.save(trained_model.state_dict(), "model_sst2_finetuned.pth")
#load the saved model
model = BertClassifier(2)
model.load_state_dict(torch.load("model_sst2_finetuned.pth"))

# Evaluate model
evaluate(model, sst2_val)