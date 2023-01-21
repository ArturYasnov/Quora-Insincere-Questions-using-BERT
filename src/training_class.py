import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.dataset import Dataset


class CFG:
    epochs = 40
    lr = 5e-6
    barch_size = 64
    scheduler_steps = [10, 25, 35]
    scheduler_gamma = 0.1

    models_dir = "/home/artur/PycharmProjects/NLP_bert_cls/models"
    experiment_name = "bert_train_minisota"


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class BertModule(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.losses_train = 0
        self.losses_valid = 0
        self.acc_train = 0
        self.acc_valid = 0
        self.len_train_data = 1
        self.len_valid_data = 1
        self.lr = CFG.lr

        self.criterion = nn.CrossEntropyLoss()

        self.model = BertClassifier()
        checkpoint = torch.load("/other/bert_3epochs.pt")
        self.model.load_state_dict(checkpoint)

    def forward(self, input_id, mask):
        outputs = self.model(input_id, mask)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=False)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=CFG.scheduler_steps, gamma=CFG.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        train_input, train_label = train_batch
        mask = train_input["attention_mask"]
        input_id = train_input["input_ids"].squeeze(1)

        outputs = self.model(input_id, mask)

        batch_loss = self.criterion(outputs, train_label.long())
        self.losses_train += batch_loss.item()

        acc = (outputs.argmax(dim=1) == train_label).sum().item()
        self.acc_train += acc

        self.log(
            "train_loss_batch",
            batch_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc_batch",
            acc / CFG.barch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return batch_loss

    def training_epoch_end(self, outs):
        self.log("train_epoch_loss", self.losses_train / self.len_train_data)
        self.log("train_epoch_acc", self.acc_train / self.len_train_data)
        self.losses_train = 0
        self.acc_train = 0

    def validation_step(self, val_batch, batch_idx):
        train_input, train_label = val_batch
        mask = train_input["attention_mask"]
        input_id = train_input["input_ids"].squeeze(1)

        outputs = self.model(input_id, mask)

        batch_loss = self.criterion(outputs, train_label.long())
        self.losses_valid += batch_loss.item()

        acc = (outputs.argmax(dim=1) == train_label).sum().item()
        self.acc_valid += acc

        self.log(
            "valid_loss_batch",
            batch_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_acc_batch",
            acc / CFG.barch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return batch_loss

    def validation_epoch_end(self, outs):
        self.log("valid_epoch_loss", self.losses_valid / self.len_valid_data)
        self.log("valid_epoch_acc", self.acc_valid / self.len_valid_data)
        self.losses_valid = 0
        self.acc_valid = 0

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    # def collate_fn(self, batch):
    #    return tuple(zip(*batch))

    def setup(self, stage=None):
        data = pd.read_csv("/home/artur/PycharmProjects/NLP_bert_cls/data/train.csv")
        cls0_sample = data[data["target"] == 0].head(2400)
        cls1_sample = data[data["target"] == 1].head(800)
        new_data = pd.concat([cls0_sample, cls1_sample])
        data = new_data.reset_index(drop=True)

        df_train, df_val = train_test_split(
            data, test_size=0.2, shuffle=True, random_state=42
        )
        df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(
            drop=True
        )

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        train_dataset, valid_dataset = Dataset(df_train, tokenizer), Dataset(
            df_val, tokenizer
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=CFG.barch_size, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=CFG.barch_size
        )

        self.len_train_data = len(df_train)
        self.len_valid_data = len(df_val)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.tr_loader = train_loader
        self.v_loader = valid_loader

    def train_dataloader(self):
        return self.tr_loader

    def val_dataloader(self):
        return self.v_loader

    def test_dataloader(self):
        return self.v_loader


if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = BertModule()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG.epochs,
        precision=16,
        gradient_clip_val=1e-1,
        fast_dev_run=False,
        profiler=None,
        # accumulate_grad_batches=1,
        callbacks=None,
    )

    trainer.fit(model)

    torch.save(model.model.state_dict(), f"{CFG.models_dir}/{CFG.experiment_name}.pth")
