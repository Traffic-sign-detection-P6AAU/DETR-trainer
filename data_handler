import os
#import torch
import torchvision
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor
#from torch.utils.data import Dataset
from torchvision import datasets
#import time
#import pandas as pd
#from torchvision import transforms as pth_transforms
from sklearn.model_selection import train_test_split
import random
import shutil

def download_dataset():
    os.system("start /wait cmd /c \"curl \"http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar\" --output training-datasets\\voc12.tar\"")
    os.chdir("training-datasets")
    os.system("start /wait cmd /c \"tar -xf voc12.tar\"")
    os.chdir("..")
    print(os.path.abspath(os.curdir))

def split_dataset(data_dir, split_ratio=0.8):
    # Create directories for train and test sets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    image_filenames = [filename for filename in os.listdir(os.path.join(data_dir, 'JPEGImages')) if filename.endswith('.jpg')]

    # shuffle liste af filenames
    random.shuffle(image_filenames)

    # Split the list of image filenames into train and test sets
    num_train = int(len(image_filenames) * split_ratio)
    train_image_filenames = image_filenames[:num_train]
    test_image_filenames = image_filenames[num_train:]

    for filename in train_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'JPEGImages', filename), os.path.join(train_dir, filename))

    for filename in test_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'JPEGImages', filename), os.path.join(test_dir, filename))

    # Split annotations files into train and test directories
    for filename in train_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'Annotations', filename.replace('.jpg', '.xml')), os.path.join(train_dir, filename.replace('.jpg', '.xml')))
    for filename in test_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'Annotations', filename.replace('.jpg', '.xml')), os.path.join(test_dir, filename.replace('.jpg', '.xml')))

    print("Number of training images:", len(train_image_filenames))
    print("Number of test images:", len(test_image_filenames))

if not os.path.exists('training-datasets\VOCdevkit'):
    download_dataset()

if not os.path.exists('training-datasets/VOCdevkit/VOC2012/test'):
    split_dataset("training-datasets/VOCdevkit/VOC2012")

'''
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

#Alternative download of data
data_handler = datasets.VOCDetection(
    root = "training-datasets",
    year = "2012",
    image_set = "train",
    transform = transform,
    target_transform = torchvision.transforms.Resize((100, 100)),
    download = True,
)

#Alternative download of data
test_data = datasets.VOCDetection(
    root = "test-datasets",
    year = "2012",
    image_set = "val",
    transform = transform,
    target_transform = torchvision.transforms.Resize((100, 100)),
    download = True,
)
'''
