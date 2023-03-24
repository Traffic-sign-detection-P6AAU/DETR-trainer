import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv
from trainer.settings import CHECKPOINT, CONFIDENCE_TRESHOLD, IOU_TRESHOLD

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
    #print(model.to(DEVICE))
    return image_processor, model

def show_img_from_pre_dataset(image_processor, model): # disable gradient calculation reduse memory 
    with torch.no_grad():

        # load image and predict
        image = cv2.imread("test_imgs/dog.jpeg")
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_TRESHOLD,
            target_sizes=target_sizes
        )[0]

    # annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)

    labels = [
        f"{model.config.id2label[class_id]} {confidence:0.2f}" 
        for _, confidence, class_id, _
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    sv.show_frame_in_notebook(frame, (16, 16))
