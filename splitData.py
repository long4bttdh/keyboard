import os
import shutil
import random

datasetDir = "/home/actvn/KeyBoard/pythonProject/.venv/wav"
trainDir = "/home/actvn/KeyBoard/pythonProject/.venv/train"
valDir = "/home/actvn/KeyBoard/pythonProject/.venv/val"
testDir = "/home/actvn/KeyBoard/pythonProject/.venv/test"

os.makedirs(trainDir, exist_ok=True)
os.makedirs(valDir, exist_ok=True)
os.makedirs(testDir, exist_ok=True)

for folder_name in os.listdir(datasetDir):
    folder_path = os.path.join(datasetDir, folder_name)
    files = os.listdir(folder_path)
    totalFiles = len(files)
    numTrain = int(0.8 * totalFiles)
    numVal = int(0.1 * totalFiles)
    numTest = totalFiles - numTrain - numVal
    trainFiles = files[:numTrain]
    valFiles = files[numTrain : numTrain + numVal]
    testFiles = files[numTrain + numVal : ]
    for filename in trainFiles:
        shutil.move(os.path.join(folder_path , filename), os.path.join(trainDir, filename))

    for filename in valFiles:
        shutil.move(os.path.join(folder_path, filename), os.path.join(valDir, filename))

    for filename in testFiles:
        shutil.move(os.path.join(folder_path, filename), os.path.join(testDir, filename))

print("Dataset has been successfully split into training, validation, and testing sets.")