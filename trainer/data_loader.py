import os
import random
import torchvision
import cv2
import supervision as sv
from torch.utils.data import DataLoader

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

    train_dataset = CocoDetection(
        image_directory_path=TRAIN_DIRECTORY,
        image_processor=image_processor,
        train=True)
    val_dataset = CocoDetection(
        image_directory_path=VAL_DIRECTORY,
        image_processor=image_processor,
        train=False)
    test_dataset = CocoDetection(
        image_directory_path=TEST_DIRECTORY,
        image_processor=image_processor,
        train=False)
    
    #print("Number of training examples:", len(train_dataset))
    #print("Number of validation examples:", len(val_dataset))
    #print("Number of test examples:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset):
    def collate_fn(batch):
        # DETR authors employ various image sizes during training, making it not possible 
        # to directly batch together images. Hence they pad the images to the biggest 
        # resolution in a given batch, and create a corresponding binary pixel_mask 
        # which indicates which pixels are real/which are padding
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=4)
    test_dataloader = DataLoader(dataset=test_dataset, collate_fn=collate_fn, batch_size=4)
    return train_dataloader, val_dataloader, test_dataloader

def show_img_from_data(train_dataset, test_dataset):
    # select random image
    image_ids = test_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = test_dataset.coco.loadImgs(image_id)[0]
    annotations = train_dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(train_dataset.root, image['file_name'])
    image = cv2.imread(image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

    # we will use id2label function for training
    labels = [
        f"{get_id2label(train_dataset)[class_id]}" 
        for _, _, class_id, _
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    sv.show_frame_in_notebook(image, (16, 16))

def get_id2label(train_dataset):
    categories = train_dataset.coco.cats
    return {k: v['name'] for k,v in categories.items()}
