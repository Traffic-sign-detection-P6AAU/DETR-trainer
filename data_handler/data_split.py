import os
import shutil
from data_handler.shared import load_json, save_json

DATASET_NAME = 'outputData'
SOURCE_DIR = 'C:/Users/Jakob/Documents/GitHub/Datasets/JPEGImages'
ANNO_NAME = '_annotations.coco.json'

def p_join(dir_1, dir_2):
    return os.path.join(dir_1, dir_2)

def split_dataset(categories_path):
    create_directories()
    print('Loading data and dividing...')
    train_labels = load_json(p_join(SOURCE_DIR, 'train.json'))
    accepted_cats = load_json(categories_path)['categories']
    accepted_cats_ids = [item['oldid'] for item in accepted_cats]
    val_labels, test_labels = divide_data(load_json(p_join(SOURCE_DIR, 'train.json')), accepted_cats_ids, accepted_cats)
    print('Finding annotations and images..')
    train_labels = find_annos_and_imgs(train_labels, accepted_cats_ids, accepted_cats)
    copy_imgs(val_labels, test_labels, train_labels)
    save_labels(val_labels, test_labels, train_labels)

def create_directories():
    train_dir = p_join(DATASET_NAME, 'train')
    test_dir =p_join(DATASET_NAME, 'test')
    val_dir = p_join(DATASET_NAME, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

def divide_data(input_data, accepted_cats_ids, accepted_cats):
    val_labels = {'annotations': input_data['annotations']}
    val_labels['images'] = input_data['images'][:len(input_data['images']) // 2]
    val_labels = find_annos_and_imgs(val_labels, accepted_cats_ids, accepted_cats)
    test_labels = {'annotations': input_data['annotations']}
    test_labels['images'] = input_data['images'][len(input_data['images']) // 2:]
    test_labels = find_annos_and_imgs(test_labels, accepted_cats_ids, accepted_cats)
    return val_labels, test_labels

def find_annos_and_imgs(labels, accepted_cats_ids, accepted_cats):
    new_annos = []
    new_imgs = []
    for anno in labels['annotations']:
        img = [img for img in labels['images'] if anno['image_id'] == img['id']]
        if len(img) == 0: continue
        if anno['category_id'] in accepted_cats_ids:
            new_annos.append(anno)
            new_imgs.append(img[0])
    result = {
        'images': new_imgs,
        'categories': accepted_cats,
        'annotations': new_annos
    }
    return make_incremental_ids(result, accepted_cats)

def copy_imgs(val_labels, test_labels, train_labels):
    print('Copying images to train, test and val directories...')
    for image in test_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join(p_join(DATASET_NAME, 'test'), image['file_name']))
    print('-Test images done')

    for image in val_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join(p_join(DATASET_NAME, 'valid'), image['file_name']))
    print('-Validation images done')

    for image in train_labels['images']:
        shutil.copy2(p_join(SOURCE_DIR, image['file_name']), p_join(p_join(DATASET_NAME, 'train'), image['file_name']))
    print('-Train images done')

def save_labels(val_labels, test_labels, train_labels):
    save_json(val_labels, p_join(p_join(DATASET_NAME, 'valid'), ANNO_NAME))
    save_json(test_labels, p_join(p_join(DATASET_NAME, 'test'), ANNO_NAME))
    save_json(train_labels, p_join(p_join(DATASET_NAME, 'train'), ANNO_NAME))

def make_incremental_ids(labels, accepted_cats):
    for item in labels['annotations']:
        new_cat_id = list(filter(lambda x: x['oldid'] == item['category_id'], accepted_cats))[0]['id']
        item['category_id'] = new_cat_id
    return labels
