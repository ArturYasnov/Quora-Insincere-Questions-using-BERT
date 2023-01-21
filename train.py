import pytorch_lightning as pl
import torch

from src.training_class import CFG, BertModule

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = BertModule()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG.epochs,
        precision=32,
        gradient_clip_val=1e-1,
        fast_dev_run=False,
        profiler=None,
        accumulate_grad_batches=4,
        callbacks=None,
    )

    trainer.fit(model)
    trainer.validate(model)
