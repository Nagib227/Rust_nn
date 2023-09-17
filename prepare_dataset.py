import os
from os import walk
import shutil


name_dataset = "flower_photos"
val_percent = 20
os.chdir("datasets")
dataset_classes = next(walk(f"{name_dataset}"), (None, None, []))[1]
os.mkdir(f"{name_dataset}_prepare")
os.mkdir(f"{name_dataset}_prepare/train")
os.mkdir(f"{name_dataset}_prepare/test")
for i in dataset_classes:
    os.mkdir(f"{name_dataset}_prepare/train/{i}")
    os.mkdir(f"{name_dataset}_prepare/test/{i}")
    files = next(walk(f"{name_dataset}/{i}"), (None, None, []))[2]
    val_files = files[:int(len(files) / 100 * val_percent) + 1]
    training_files = files[int(len(files) / 100 * val_percent) + 1:]
    for j in training_files:
        shutil.copy(f"{name_dataset}/{i}/{j}", f"{name_dataset}_prepare/train/{i}/{j}")
        
    for j in val_files:
        shutil.copy(f"{name_dataset}/{i}/{j}", f"{name_dataset}_prepare/test/{i}/{j}")
