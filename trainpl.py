import torch
import pytorch_lightning as pl

class MeshGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage = None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


class MeshGraphNets(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

def main():
    data_dir = "cloth" # TODO
    trainer = pl.Trainer(accelerator="gpu", strategy="ddp")
    dm = MeshGraphDataModule(data_dir)
    model = MeshGraphNets()
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()