# import keras
# import numpy as np
# import glob
# import gc
# import matplotlib.pyplot as plt
# import librosa
# import pandas as pd
# from keras.src.layers import BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Reshape
# from tensorflow.keras import optimizers
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# import torch
#
# def append_ext(fn):
#   return fn.replace(".wav",".png")
#
# train_data_path = "/home/actvn/KeyBoard/pythonProject/.venv/train_mel"
# val_data_path = "/home/actvn/KeyBoard/pythonProject/.venv/val_mel"
# test_data_path = "/home/actvn/KeyBoard/pythonProject/.venv/test_mel"
#
# traindf = pd.read_csv("train_dataset.csv")
# valdf = pd.read_csv("val_dataset.csv")
# testdf = pd.read_csv("test_dataset.csv")
# traindf["slice_file_name"]=traindf["slice_file_name"].apply(append_ext)
# valdf["slice_file_name"] = valdf["slice_file_name"].apply(append_ext)
# testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)
#
# datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
#
# train_generator=datagen.flow_from_dataframe(
#   dataframe=traindf,
#   directory=train_data_path,
#   x_col="slice_file_name",
#   y_col="class",
#   subnet="training",
#   batch_size=32,
#   seed=42,
#   shuffle=True,
#   class_mode="categorical",
#   target_size=(64,64)
# )
#
# valid_generator=datagen.flow_from_dataframe(
#   dataframe=valdf,
#   directory=val_data_path,
#   x_col="slice_file_name",
#   y_col="class",
#   subnet="validation",
#   batch_size=32,
#   seed=42,
#   shuffle=True,
#   class_mode="categorical",
#   target_size=(64,64)
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# #
# # # Xây dựng mô hình
# model = Sequential()
# model.add(Conv2D(32,(3,3),padding='same',input_shape=(64,64,3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3),padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(128,(3,3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Reshape((1,512)))
# # model.add(LSTM(512, dropout=0.2))
# # model.add(LSTM(512, dropout=0.2))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(36,activation='softmax'))
# model.build()
#
# # Compile model
# optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", "auc"])
# model.summary()
#
# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID=train_generator.n//valid_generator.batch_size
#
# history = model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=STEP_SIZE_TRAIN,  # Số lượng bước trong mỗi epoch (là tổng của STEP và SIZE_TRAIN)
#     validation_data=valid_generator,
#     validation_steps=STEP_SIZE_VALID,  # Số lượng bước trong quá trình validation
#     epochs=20,
#     verbose=1
# )
#
# # Move to GPU if available
# if tf.config.list_physical_devices('GPU'):
#   tf.keras.backend.set_floatx('float32')  # Ensure the model uses float32
#   print("GPU is available.")
# else:
#   print("Using CPU.")
#
# # Lấy các giá trị mất mát từ lịch sử đào tạo
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# # Vẽ biểu đồ độ mất mát
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Assume we have the following paths. Depend on your system, it could vary
TRAIN_DIR = "/home/actvn/KeyBoard/pythonProject/.venv/train_mel"
VAL_DIR = "/home/actvn/KeyBoard/pythonProject/.venv/val_mel"
TEST_DIR = "/home/actvn/KeyBoard/pythonProject/.venv/test_mel"
MODEL_PATH = "/home/actvn/KeyBoard/pythonProject/.venv/model"

os.makedirs(MODEL_PATH, exist_ok=True)

# The following class help transform our input into mel-spectrogram
# class ToMelSpectrogram:
#     def __call__(self, samples):
#         return librosa.feature.melspectrogram(samples)


# This class is to load audio data and apply the transformation
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


def train():
    # We will use the transformation to convert the audio into Mel spectrogram
    transform = Compose([transforms.ConvertImageDtype(torch.float),transforms.ToTensor()])

    train_set = AudioDataset(TRAIN_DIR, transform=transform)
    val_set = AudioDataset(VAL_DIR, transform=transform)
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=8, shuffle=True)

    model = CoAtNet()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            # print(labels)
            # labels = np.array(labels)
            # input_labels = torch.Tensor(labels)
            # labels = input_labels.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in val_loader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                validation_loss = correct/total
                print(f"Validation Accuracy: {validation_loss}")
                if early_stopper.early_stop(validation_loss):
                    break

    torch.save(model.state_dict(), "/home/actvn/KeyBoard/pythonProject/.venv/model/model_test.pt")

def main():
    train()

if __name__ == "__main__":
    main()