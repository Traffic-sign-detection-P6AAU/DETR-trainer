import os
import shutil
import json

def split_dataset(source_dir):
    # Create directories for train, test and val sets
    target_dir = os.path.join('Datasets/traffic_signs')
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    val_dir = os.path.join(target_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Read image filenames and annotations from source dir
    test_labels = json.load(open(os.path.join(source_dir, 'test.json'), 'r'))
    train_labels = json.load(open(os.path.join(source_dir, 'train.json'), 'r'))
    accepted_cats = json.load(open(os.path.join(source_dir, 'accepted_cats.json'), 'r'))['categories']
    accepted_cats_ids = [item['id'] for item in accepted_cats]


    # Split test annotations into new test and validation json objects
    val_labels = {'images': [], 'annotations': test_labels['annotations']}
    val_labels['images'] = test_labels['images'][:int(len(test_labels["images"]) / 2)]
    new_test_labels = {'images': [], 'annotations': test_labels['annotations']}
    new_test_labels['images'] = test_labels['images'][int(len(test_labels["images"]) / 2):]


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
