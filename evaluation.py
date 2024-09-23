import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import os
from torchvision import transforms, utils, datasets
from torchvision.io import read_image
import torch
from torch import nn
from sklearn.metrics import f1_score
import time

model_path = "/home/actvn/KeyBoard/pythonProject/.venv/model/model_test.pt"
TEST_DIR = "/home/actvn/KeyBoard/pythonProject/.venv/test_mel"

class CoAtNet(nn.Module):
    def __init__(self, num_classes=40):
        super(CoAtNet, self).__init__()

        # Convolutional part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Transformer part
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Linear classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)

        # Flattening
        x = x.view(x.size(0), -1, x.size(1))

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Max pooling over time
        x, _ = torch.max(x, dim=1)

        # Classifier
        x = self.fc(x)
        return x
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(self.data_dir)
        self.labels = [file_name.split("_")[0] for file_name in self.file_list]
        # Create a dictionary for label encoding
        self.label_dict = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.encoded_labels = [self.label_dict[label] for label in self.labels]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.listdir(self.data_dir)
        image = read_image(self.data_dir + "/"+ img_path[idx])
        image = image / 255.0
        label = self.file_list[idx].split("_")[0]  # Assuming the file name is 'label_otherInfo.wav'
        encoded_label = self.label_dict[label]
        return image, encoded_label
startTime = time.time()
transform = Compose([transforms.ConvertImageDtype(torch.float),transforms.ToTensor()])
test_set = AudioDataset(TEST_DIR, transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=True)
model = CoAtNet()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():  # No gradient computation needed for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # Get the class with the highest score
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Step 4: Calculate the F1 score
f1 = f1_score(all_labels, all_preds, average='weighted')  # or 'macro', 'micro' as needed
endTime = time.time()
print(f'F1 Score: {f1:.4f}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_parameters = count_parameters(model)
print(f'Number of parameters: {num_parameters}')

print(f"Time of evaluation: {endTime -startTime}")