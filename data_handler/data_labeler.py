import os
import cv2
from data_handler.shared import load_json, save_json

IMGS_PATH = '../Datasets/mix'
CATEGORIES_PATH = 'data_handler/accepted_categories.json'
MERGE_ANNO_PATH = 'outputData/train/_annotations.coco.json'
SAVE_ANNO_PATH = '_annotations.coco.json'
BBOX_MARGIN = 5

def extend_annotations():
    categories = load_json(CATEGORIES_PATH)['categories']
    imgs, annos = make_imgs_annos(categories)
    merge_dataset = load_json(MERGE_ANNO_PATH)
    merge_dataset['images'].extend(imgs)
    merge_dataset['annotations'].extend(annos)
    merge_dataset['categories'] = (categories)
    save_json(merge_dataset, SAVE_ANNO_PATH)

def save_annotations():
    categories = load_json(CATEGORIES_PATH)['categories']
    imgs, annos = make_imgs_annos(categories)
    annotations = {
        'images': imgs,
        'categories': categories,
        'annotations': annos
    }
    save_json(annotations, SAVE_ANNO_PATH)


def make_imgs_annos(categories):
    img_id = 20000
    images = []
    annotations = []
    for directory in os.listdir(IMGS_PATH):
        dtr_path = os.path.join(IMGS_PATH, directory)
        category_id = get_id_from_dir(categories, directory)
        for file in os.listdir(dtr_path):
            #if not file.endswith('.jpg'): continue
            img_size = cv2.imread(os.path.join(dtr_path, file)).shape
            images.append(make_img(file, img_id, img_size))
            bbox = get_bbox(img_size)
            area = img_size[0] * img_size[1]
            annotations.append(make_anno(img_id, category_id, bbox, area))
            img_id += 1
    return images, annotations

def get_bbox(img_size):
    left = BBOX_MARGIN
    top = BBOX_MARGIN
    right = img_size[1] - (BBOX_MARGIN * 2)
    bottom = img_size[0] - (BBOX_MARGIN * 2)
    return [left, top, right, bottom]

def darw_bbox(path):
    image = cv2.imread(path)
    left = 35
    top = 24
    right = 108
    bottom = 96
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    cv2.imwrite('example_bbox.jpg', image)

def get_id_from_dir(categories, directory):
    for cat in categories:
        if cat['foldername'] == directory:
            return cat['id']

def make_img(file, id, img_size):
    return {
        'id': id,
        'height': img_size[0],
        'width': img_size[1],
        'file_name': file
    }


def make_anno(image_id, category_id, bbox, area):
    return {
        'id': image_id,
        'area': area,
        'bbox': bbox,
        'category_id': category_id,
        'image_id': image_id,
        'ignore': False,
        'iscrowd': 0
    }
