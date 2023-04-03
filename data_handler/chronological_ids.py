from data_handler.shared import load_json, save_json

DATASET_PATH = "_annotations.coco.json"

def set_ids():
    dataset = load_json(DATASET_PATH)
