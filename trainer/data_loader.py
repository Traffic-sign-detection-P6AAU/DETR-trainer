import os
import random
import torchvision
import cv2
import supervision as sv

# settings
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join("JPEGImages", "train")
VAL_DIRECTORY = os.path.join("JPEGImages", "valid")
TEST_DIRECTORY = os.path.join("JPEGImages", "test")

def load_datasets(image_processor):
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(
            self, 
            image_directory_path: str, 
            image_processor, 
            train: bool = True
        ):
            annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
            super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
            self.image_processor = image_processor

        def __getitem__(self, idx):
            images, annotations = super(CocoDetection, self).__getitem__(idx)        
            image_id = self.ids[idx]
            annotations = {'image_id': image_id, 'annotations': annotations}
            encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            target = encoding["labels"][0]

            return pixel_values, target


    TRAIN_DATASET = CocoDetection(
        image_directory_path=TRAIN_DIRECTORY,
        image_processor=image_processor,
        train=True)
    VAL_DATASET = CocoDetection(
        image_directory_path=VAL_DIRECTORY,
        image_processor=image_processor,
        train=False)
    TEST_DATASET = CocoDetection(
        image_directory_path=TEST_DIRECTORY,
        image_processor=image_processor,
        train=False)

    print("Number of training examples:", len(TRAIN_DATASET))
    print("Number of validation examples:", len(VAL_DATASET))
    print("Number of test examples:", len(TEST_DATASET))

def show_img_from_data():
    # select random image
    image_ids = TRAIN_DATASET.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
    annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
    image = cv2.imread(image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

    # we will use id2label function for training
    categories = TRAIN_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}

    labels = [
        f"{id2label[class_id]}" 
        for _, _, class_id, _
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    sv.show_frame_in_notebook(image, (16, 16))
