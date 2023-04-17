import torch
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
from trainer.model import get_img_processor

def evaluate_on_test_data(model, test_dataset, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=['bbox'])

    print('Running evaluation...')

    for idx, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target['orig_size'] for target in labels], dim=0)
            image_processor = get_img_processor()
            results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

            predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
            predictions = prepare_for_coco_detection(predictions)
            evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    with open('evaluation.txt', 'w', encoding='utf-8') as f:
        f.write(evaluator.get_results())


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

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
