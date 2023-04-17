import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from trainer.model import Detr
from trainer.settings import MAX_EPOCHS, LEARN_RATE, LEARN_RATE_BACKBONE, ACCUMULATE_GRAD_BATCHES, WEIGHT_DECAY

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
def start_training(train_dataloader, val_dataloader, id2label):
    model = Detr(lr=LEARN_RATE,
                 lr_backbone=LEARN_RATE_BACKBONE,
                 weight_decay=WEIGHT_DECAY,
                 train_load=train_dataloader,
                 val_load=val_dataloader,
                 id2label=id2label)

    batch = next(iter(train_dataloader))
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    logger = TensorBoardLogger('tb_logs', name='P6_model')
    trainer = Trainer(devices=1,
                      accelerator='gpu',
                      logger=logger,
                      max_epochs=MAX_EPOCHS,
                      gradient_clip_val=0.1,
                      accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
                      log_every_n_steps=1)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model)
    # Create a TensorBoard callback
    return model.to(DEVICE)
