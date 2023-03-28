import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from trainer.model import Detr
from trainer.settings import MAX_EPOCHS

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
def start_training(train_dataloader, val_dataloader, id2label):
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_load=train_dataloader, val_load=val_dataloader, id2label=id2label)

    batch = next(iter(train_dataloader))
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    # settings
    # pytorch_lightning < 2.0.0
    # trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
    # pytorch_lightning >= 2.0.0
    logger = TensorBoardLogger("tb_logs", name="P6_model")
    trainer = Trainer(devices=1, accelerator="gpu", logger=logger, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=1)
    trainer.fit(model)
    # Create a TensorBoard callback
    return model.to(DEVICE)
