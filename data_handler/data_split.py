import os
import random
import shutil
import json


def split_dataset(source_dir):
    # Create directories for train, test and val sets
    target_dir = os.path.join('Datasets')
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    val_dir = os.path.join(target_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Read image filenames and annotations from source dir
    all_image_filenames = [filename for filename in os.listdir(source_dir) if filename.endswith('.jpg')]
    test_labels = json.load(open(os.path.join(source_dir, 'test.json'), 'r'))
    train_labels = json.load(open(os.path.join(source_dir, 'train.json'), 'r'))

    # Find all image filenames for test and train sets
    test_image_filenames = []
    for items in test_labels['images']:
        test_image_filenames.append(items['file_name'])
    train_image_filenames = []
    for items in train_labels['images']:
        train_image_filenames.append(items['file_name'])

    # Split test img filenames into test and validation sets
    val_image_filenames = test_image_filenames[:int(len(test_image_filenames) / 2)]
    test_image_filenames = test_image_filenames[int(len(test_image_filenames) / 2):]

    # Split test annotations into new test and validation json objects
    val_labels = {'images': [], 'annotations': []}
    val_labels['images'] = test_labels['images'][:int(len(test_labels["images"]) / 2)]

    max_img_id = max(item['id'] for item in val_labels['images'])
    min_img_id = min(item['id'] for item in val_labels['images'])
    for item in test_labels['annotations']:
        if item['image_id'] <= max_img_id and item['image_id'] >= min_img_id:
            val_labels['annotations'].append(item)

    new_test_labels = {'images': [], 'annotations': []}
    new_test_labels['images'] = test_labels['images'][int(len(test_labels["images"]) / 2):]

    max_img_id = max(item['id'] for item in new_test_labels['images'])
    min_img_id = min(item['id'] for item in new_test_labels['images'])
    for item in test_labels['annotations']:
        if item['image_id'] <= max_img_id and item['image_id'] >= min_img_id:
            new_test_labels['annotations'].append(item)
    
    # Copy images from source dir to train, test and val dirs
    print("Copying images to train, test and val directories...")
    for filename in test_image_filenames:
        shutil.copy2(os.path.join(source_dir, filename), os.path.join(test_dir, filename))
    print("-Test images done")
    for filename in val_image_filenames:
        shutil.copy2(os.path.join(source_dir, filename), os.path.join(val_dir, filename))
    print("-Validation images done")
    for filename in train_image_filenames:
        shutil.copy2(os.path.join(source_dir, filename), os.path.join(train_dir, filename))
    print("-Train images done")
    
    # Write/copy json annotation files
    print("Writing json annotation files...")
    with open(os.path.join(test_dir, 'test.json'), "w") as outfile1:
        json.dump(new_test_labels, outfile1)
    with open(os.path.join(val_dir, 'val.json'), "w") as outfile2:
        json.dump(val_labels, outfile2)
    shutil.copy2(os.path.join(source_dir, 'train.json'), os.path.join(train_dir, 'train.json'))

    print("Number of test images:", len(test_image_filenames))
    print("Number of validation images:", len(val_image_filenames))
    print("Number of training images:", len(train_image_filenames))
