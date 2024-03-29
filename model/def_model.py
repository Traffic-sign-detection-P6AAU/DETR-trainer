import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from settings import CHECKPOINT, MODEL_PATH

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, train_load, val_load, id2label):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        self.train_load = train_load
        self.val_load = val_load
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log('training_loss', loss)
        for k,v in loss_dict.items():
            self.log('train_' + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log('validation/loss', loss)
        for k, v in loss_dict.items():
            self.log('validation_' + k, v.item())

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log('test/loss', loss)
        for k, v in loss_dict.items():
            self.log('test_' + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                'params': [p for n, p in self.named_parameters() if 'backbone' not in n and p.requires_grad]},
            {
                'params': [p for n, p in self.named_parameters() if 'backbone' in n and p.requires_grad],
                'lr': self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
    
    def train_dataloader(self):
        return self.train_load

    def val_dataloader(self):
        return self.val_load

def save_model(model):
    model.model.save_pretrained(MODEL_PATH)

def get_model(path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetrForObjectDetection.from_pretrained(path)
    return model.to(device)

def get_img_processor():
    return DetrImageProcessor.from_pretrained(CHECKPOINT)
