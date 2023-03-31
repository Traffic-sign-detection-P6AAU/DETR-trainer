import os
import json
import cv2

IMGS_PATH = "C:/Users/Jakob/Downloads/mix/mixed-data/"
CATEGORIES_PATH = "data_handler/accepted_categories.json"
BBOX_MARGIN = 5

def save_annotations():
    imgs, annos = make_imgs_annos()
    annotations = {
        "images": imgs,
        "annotations": annos
    }
    save_json(annotations)

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data):
    with open('_annotations.coco.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

def make_imgs_annos():
    img_id = 0
    images = []
    annotations = []
    categories = load_json(CATEGORIES_PATH)["categories"]
    for directory in os.listdir(IMGS_PATH):
        dtr_path = os.path.join(IMGS_PATH, directory)
        category_id = get_id_from_dir(categories, directory)
        for file in os.listdir(dtr_path):
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
    right = img_size[1] - BBOX_MARGIN
    bottom = img_size[0] - BBOX_MARGIN
    return [left, top, right, bottom]

def darw_bbox(path):
    image = cv2.imread(path)
    height, width, c = image.shape
    left = 5
    top = 5
    right = width - 5
    bottom = height - 5
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    cv2.imwrite('example_bbox.jpg', image)

def get_id_from_dir(categories, directory):
    for cat in categories:
        if cat["foldername"] == directory:
            return cat["id"]

def make_img(file, id, img_size):
    return {
        "id": id,
        "height": img_size[0],
        "width": img_size[1],
        "file_name": file
    }


def make_anno(image_id, category_id, bbox, area):
    return {
        "id": image_id,
        "area": area,
        "bbox": bbox,
        "category_id": category_id,
        "image_id": image_id,
        "ignore": False,
        "iscrowd": 0
    }

save_annotations()
