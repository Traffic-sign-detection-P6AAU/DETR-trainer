from pytorch_lightning import Trainer
from model import Detr
from trainer.settings import MAX_EPOCHS
    
def start_training():
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    batch = next(iter(TRAIN_DATALOADER))
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    # settings
    # pytorch_lightning < 2.0.0
    # trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
    # pytorch_lightning >= 2.0.0
    trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
    trainer.fit(model)