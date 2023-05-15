from argparse import Namespace
from data_handler.img_show import show_img_w_prediction
from data_handler.data_loader import load_datasets, get_dataloaders, get_id2label
from model.train import start_training
from model.def_model import save_model, get_model, get_img_processor
from settings import MODEL_PATH, CHECKPOINT
from evaluation.evaluate_test_data import evaluate_on_test_data
from transformers import DetrForObjectDetection
import torch
import detr

CATEGORIES_PATH = 'data_handler/categories.json'

def main():
    print('---Menu list---')
    print('Type: 1 to train the model')
    print('Type: 2 to use the model')
    choice = input()
    if choice == '1':
        image_processor = get_img_processor()
        train_dataset, val_dataset, test_dataset = load_datasets(image_processor)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset)
        trained_model = start_training(train_dataloader, val_dataloader, test_dataloader, get_id2label(train_dataset))
        save_model(trained_model)
        evaluate_on_test_data(trained_model, test_dataloader, test_dataset)
    elif choice == '2':
        args = Namespace(lr=0.0001, lr_backbone=1e-05, batch_size=2, weight_decay=0.0001, epochs=300, lr_drop=200, clip_max_norm=0.1, frozen_weights=None, backbone='resnet50', dilation=False, position_embedding='sine', enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=256, dropout=0.1, nheads=8, num_queries=100, pre_norm=False, masks=False, aux_loss=True, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, mask_loss_coef=1, dice_loss_coef=1, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, dataset_file='coco', coco_path=None, coco_panoptic_path=None, remove_difficult=False, output_dir='', device='cuda', seed=42, resume='', start_epoch=0, eval=False, num_workers=2, world_size=1, dist_url='env://', distributed=False)
        image_processor = get_img_processor()
        model, criterion, postprocessors = detr.build(args)
        state_dist = torch.load('checkpoint0299.pth', map_location='cpu')
        model.load_state_dict(state_dict=state_dist['model'])
        show_img_w_prediction(image_processor, model, CATEGORIES_PATH)
    else:
        print('Input was not 1 or 2.')

if __name__ == '__main__':
    main()
