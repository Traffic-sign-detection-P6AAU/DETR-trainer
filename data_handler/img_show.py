import random
import os
import torch
import cv2
import supervision as sv
from settings import CONFIDENCE_TRESHOLD, IOU_TRESHOLD
from data_handler.shared import load_json
from data_handler.data_loader import get_id2label
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

CATEGORIES_PATH = 'data_handler/categories.json'

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
def plot_results(pil_img, prob, boxes):
    categories = load_json(CATEGORIES_PATH)['categories']
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{categories[cl]["name"]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def show_img_w_prediction(image_processor, model, CATE): # disable gradient calculation reduse memory 
    transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    device = torch.device('cpu')

    with torch.no_grad():

        # load image and predict

        im = Image.open('Datasets/test3.jpg')
        inputs = image_processor(images=im, return_tensors='pt').to(device)
        img = transform(im).unsqueeze(0)
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        plot_results(im, probas[keep], bboxes_scaled)
        return

        # post-process
        target_sizes = torch.tensor([im.shape[:2]]).to(device)
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
    frame = box_annotator.annotate(scene=im, detections=detections, labels=labels)

    sv.show_frame_in_notebook(frame, (16, 16))

def show_model_prediction(test_dataset, image_processor, model):
    device = torch.device('cpu')
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
