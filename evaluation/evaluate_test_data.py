import torch
#from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
import os
import cv2
import supervision as sv
from model.def_model import get_img_processor
from settings import CONFIDENCE_TRESHOLD, IOU_TRESHOLD

def evaluate_accuracy(model, test_dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_processor = get_img_processor()
    num_correct = 0
    errors = 0
    i = 0
    print('Evaluating on test data...')
    for image in test_dataset.coco.imgs.values():
        print('Progress: ', i, '/', len(test_dataset.coco.imgs.values()))
        image_path = os.path.join(test_dataset.root, image['file_name'])
        annotations = test_dataset.coco.imgToAnns[image['id']]
        image = cv2.imread(image_path)
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
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)

        res = detections.class_id
        gt = [ann['category_id'] for ann in annotations]
        if len(res) != len(gt):
            errors += 1
        elif all(res == gt):
            num_correct += 1
        else:
            errors += 1
        i += 1
    print('num_correct: ', num_correct, 'errors: ', errors)
    print('accuracy: ', num_correct / (num_correct + errors))
    
def evaluate_on_test_data(model, test_dataloader, test_dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=['bbox'])
    print('Running evaluation...')
    image_processor = get_img_processor()
    for idx, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target['orig_size'] for target in labels], dim=0)
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        if len(predictions) <= 0:
            print('No predictions')
            continue
        evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction['boxes']
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction['scores'].tolist()
        labels = prediction['labels'].tolist()
        coco_results.extend(
            [
                {
                    'image_id': original_id,
                    'category_id': labels[k],
                    'bbox': box,
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results
