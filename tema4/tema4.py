import pickle
import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import Tensor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 64
INPUT_SIZE = 784
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
OUTPUT_SIZE = 10
NUM_EPOCHS = 30
DROPOUT_PROB = 0.1
LEARNING_RATE = 0.001

# Augumentations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

class ExtendedMNISTDataset(Dataset):
    # def __init__(self, root: str = "/kaggle/input/fii-nn-2025-homework-4", train: bool = True):
    def __init__(self, root: str = "tema4/input", train: bool = True):
        self.train = train
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self, ) -> int:
        return len(self.data)

    def __getitem__(self, i : int):
        image, label = self.data[i]
        image = image.reshape(28, 28)
        if self.train:
            image = train_transform(image)
        else:
            image = torch.as_tensor(image, dtype=torch.float32) / 255.0
        image = image.flatten()
        label = torch.as_tensor(label, dtype=torch.long)
        return image, label

# DataLoaders
dataloader_train = DataLoader(ExtendedMNISTDataset(train=True), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(ExtendedMNISTDataset(train=False), batch_size=BATCH_SIZE, shuffle=False)

# Model
class MyModel(nn.Module):
    def __init__(self,
                 input_size: int = INPUT_SIZE,
                 hidden_size_1: int = HIDDEN_SIZE_1,
                 hidden_size_2: int = HIDDEN_SIZE_2,
                 output_size: int = OUTPUT_SIZE,
                 dropout_prob: float = DROPOUT_PROB
                ) -> None:
        super(MyModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output_layer = nn.Linear(hidden_size_2, output_size)

        self.bn_1 = nn.BatchNorm1d(hidden_size_1)
        self.bn_2 = nn.BatchNorm1d(hidden_size_2)

        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.layer_1(x))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = torch.relu(self.layer_2(x))
        x = self.bn_2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

model = MyModel()

#Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(device)
model = model.to(device)

#Optimizer
optimizer =  torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)

#Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#Criterion
criterion = nn.CrossEntropyLoss()

#Accuracy
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) 
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    accuracy = (correct_predictions / total_samples) * 100
    model.train()
    return accuracy

#Training function
def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    mean_loss = 0.0
    for data, labels in train_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        mean_loss += loss.item()
    mean_loss /= len(train_dataloader)
    return mean_loss

# Training loop
def main(model, train_dataloader, optimizer, criterion, device, scheduler, num_epochs = NUM_EPOCHS):
    with tqdm(tuple(range(num_epochs)), unit="epoch") as tbar:
        for epoch in tbar:
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            train_accuracy = evaluate_accuracy(model, train_dataloader, device)
            scheduler.step()
            tbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

main(model, dataloader_train, optimizer, criterion, device, scheduler, NUM_EPOCHS)

# Predict function
def predict(model, test_dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_dataloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

predictions = predict(model, dataloader_test, device)

# This is how you prepare a submission for the competition
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("tema4/submission.csv", index=False)