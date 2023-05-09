import random
import os
import torch
import cv2
import supervision as sv
from settings import CONFIDENCE_TRESHOLD, IOU_TRESHOLD
from data_handler.shared import load_json
from data_handler.data_loader import get_id2label

def show_img_w_prediction(image_processor, model, categories_path): # disable gradient calculation reduse memory 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        # load image and predict
        image = cv2.imread('../Datasets/google_maps/image3.png')
        inputs = image_processor(images=image, return_tensors='pt').to(device)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_TRESHOLD,
            target_sizes=target_sizes
        )[0]

    # annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)
    categories = load_json(categories_path)['categories']
    labels = []
    for _, confidence, class_id, _ in detections:
        class_name = categories[class_id]['name']
        labels.append(f'{class_name} {confidence:0.2f}' )

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    sv.show_frame_in_notebook(frame, (16, 16))

def show_model_prediction(test_dataset, image_processor, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # utils
    categories = test_dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()

    # select random image
    image_ids = test_dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = test_dataset.coco.loadImgs(image_id)[0]
    annotations = test_dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(test_dataset.root, image['file_name'])
    image = cv2.imread(image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels = [f'{id2label[class_id]}' for _, _, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    print('ground truth')

    sv.show_frame_in_notebook(frame, (16, 16))

    # inference
    with torch.no_grad():

        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(device)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs, 
            threshold=CONFIDENCE_TRESHOLD, 
            target_sizes=target_sizes
        )[0]

    # annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
    labels = [f'{id2label[class_id]} {confidence:.2f}' for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    print('detections')
    sv.show_frame_in_notebook(frame, (16, 16))

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
        f'{get_id2label(train_dataset)[class_id]}' 
        for _, _, class_id, _
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    sv.show_frame_in_notebook(image, (16, 16))
