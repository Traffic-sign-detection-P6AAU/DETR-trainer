import os
import shutil
from shared import load_json, save_json

CATEGORIES_PATH = 'data_handler/accepted_categories.json'
DATASET_NAME = 'uniLoginData'
SOURCE_DIR = 'uniData'
MERGE_DATASET_PATH = "_annotations.coco.json"

def p_join(dir_1, dir_2):
    return os.path.join(dir_1, dir_2)

def split_dataset():
    train_labels = load_json('train.json')
    accepted_cats = load_json(CATEGORIES_PATH)['categories']
    accepted_cats_ids = [item['id'] for item in accepted_cats]
    val_labels, test_labels = divide_data(load_json('test.json'), accepted_cats_ids)
    train_labels = find_annos_and_imgs(train_labels, accepted_cats_ids)
    remove_imgs(val_labels, test_labels, train_labels)
    save_datasets(val_labels, test_labels, train_labels)

def create_directories():
    train_dir = p_join(DATASET_NAME, 'train')
    test_dir =p_join(DATASET_NAME, 'test')
    val_dir = p_join(DATASET_NAME, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
def divide_data(input_data, accepted_cats_ids):
    val_labels = {'annotations': input_data['annotations']}
    val_labels['images'] = input_data['images'][:len(input_data['images'] // 2)]
    val_labels = find_annos_and_imgs(val_labels, accepted_cats_ids)
    test_labels = {'annotations': input_data['annotations']}
    test_labels['images'] = input_data['images'][len(input_data['images']) // 2:]
    test_labels = find_annos_and_imgs(test_labels, accepted_cats_ids)
    return val_labels, test_labels

def find_annos_and_imgs(labels, accepted_cats_ids):
    new_annos = []
    new_imgs = []
    for anno in labels['annotations']:
        img_id = [img['id'] for img in labels['images'] if anno['image_id'] == img['id']]
        if anno['category_id'] in accepted_cats_ids:
            new_annos.append(anno)
            new_imgs.append(labels['images'][img_id])
    return {
        'annotations': new_annos,
        'images': new_imgs
    }

def remove_imgs(val_labels, test_labels, train_labels):
    print("Copying images to train, test and val directories...")
    for image in test_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join('test', image['file_name']))
    print("-Test images done")

    for image in val_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join('valid', image['file_name']))
    print("-Validation images done")

    for image in train_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join('train', image['file_name']))
    print("-Train images done")

def save_datasets(val_labels, test_labels, train_labels):
    save_json(val_labels, MERGE_DATASET_PATH)
    save_json(test_labels, MERGE_DATASET_PATH)
    save_json(train_labels, MERGE_DATASET_PATH)

"""
def split_dataset_old(source_dir):
    # Create directories for train, test and val sets
    target_dir = os.path.join('Datasets/traffic_signs')
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    val_dir = os.path.join(target_dir, 'valid')


    # Read image filenames and annotations from source dir
    train_labels = json.load(open(os.path.join(source_dir, 'train.json'), 'r', encoding='utf-8'))
    accepted_cats = json.load(open(CATEGORIES_PATH, 'r', encoding='uft-8'))['categories']


    # Split test annotations into new test and validation json objects



    # Make validation json
    new_val_annotations = []
    for item in val_labels['annotations']:
        if item['image_id'] in [item['id'] for item in val_labels['images']] and item['category_id'] in accepted_cats_ids:
            new_val_annotations.append(item)
    val_labels['images'] = list(filter(lambda x: x['id'] in [item['image_id'] for item in new_val_annotations], val_labels['images']))
    val_labels['annotations'] = new_val_annotations


    # Make test json
    new_test_annotations = []
    for item in new_test_labels['annotations']:
        if item['image_id'] in [item['id'] for item in new_test_labels['images']] and item['category_id'] in accepted_cats_ids:
            new_test_annotations.append(item)
    new_test_labels['images'] = list(filter(lambda x: x['id'] in [item['image_id'] for item in new_test_annotations], new_test_labels['images']))
    new_test_labels['annotations'] = new_test_annotations

    # Make train json
    new_train_labels = {'images': [], 'annotations': []}
    for item in train_labels['annotations']:
        if item['category_id'] in accepted_cats_ids:
            new_train_labels['annotations'].append(item)
    new_train_labels['images'] = list(filter(lambda x: x['id'] in [item['image_id'] for item in new_train_labels['annotations']], train_labels['images']))
    
    # Copy images from source dir to train, test and val dirs
    print("Copying images to train, test and val directories...")
    for image in new_test_labels['images']:
        shutil.copy2(os.path.join(source_dir, image['file_name']), os.path.join(test_dir, image['file_name']))
    print("-Test images done")

    for image in val_labels['images']:
        shutil.copy2(os.path.join(source_dir, image['file_name']), os.path.join(val_dir, image['file_name']))
    print("-Validation images done")

    for image in new_train_labels['images']:
        shutil.copy2(os.path.join(source_dir, image['file_name']), os.path.join(train_dir, image['file_name']))
    print("-Train images done")
    
    # Write/copy json annotation files
    print("Writing json annotation files...")
    with open(os.path.join(test_dir, '_annotations.coco.json'), "w") as outfile1:
        json.dump(new_test_labels, outfile1)
    with open(os.path.join(val_dir, '_annotations.coco.json'), "w") as outfile2:
        json.dump(val_labels, outfile2)
    with open(os.path.join(train_dir, '_annotations.coco.json'), "w") as outfile3:
        json.dump(new_train_labels, outfile3)

    print("Number of test images:", len(new_test_labels['images']))
    print("Number of validation images:", len(val_labels['images']))
    print("Number of training images:", len(new_train_labels['images']))
"""