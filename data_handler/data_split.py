import os
import random
import shutil
import json


def split_dataset():
    # Create directories for train and test sets
    train_dir = os.path.join('train')
    test_dir = os.path.join('test')
    val_dir = os.path.join('valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    target_dir = os.path.join('Results')

    all_image_filenames = [filename for filename in os.listdir(os.path.join('JPEGImages')) if filename.endswith('.jpg')]
    test_labels = json.load(open('JPEGImages/test.json', 'r'))
    test_image_filenames = [filename for filename in test_labels['images']]
    

    # Split the list of image filenames into train, test and val sets
    num_train = int(len(all_image_filenames) * train_ratio)
    num_test = num_train + int(len(all_image_filenames) * (1 - train_ratio) / 2)
    print(num_train, num_test)
    train_image_filenames = all_image_filenames[:num_train]
    test_image_filenames = all_image_filenames[num_train:num_test]
    val_image_filenames = all_image_filenames[num_test:]

    for filename in train_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'JPEGImages', filename), os.path.join(train_dir, filename))
    for filename in test_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'JPEGImages', filename), os.path.join(test_dir, filename))
    for filename in test_image_filenames:
        shutil.copy2(os.path.join(data_dir, 'JPEGImages', filename), os.path.join(test_dir, filename))

    # # Split annotations files into train and test directories
    # for filename in train_image_filenames:
    #     shutil.copy2(os.path.join(data_dir, 'Annotations', filename.replace('.jpg', '.xml')), os.path.join(train_dir, filename.replace('.jpg', '.xml')))
    # for filename in test_image_filenames:
    #     shutil.copy2(os.path.join(data_dir, 'Annotations', filename.replace('.jpg', '.xml')), os.path.join(test_dir, filename.replace('.jpg', '.xml')))

    print("Number of training images:", len(train_image_filenames))
    print("Number of test images:", len(test_image_filenames))
    print("Number of validation images:", len(val_image_filenames))
