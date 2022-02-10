from typing import List, Optional, Tuple

from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.datasets.multiconer import load_multiconer


class MultiCoNERDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        task_module: TaskModule,
        name: str = "en",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.task_module = task_module
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[List[TaskEncoding]] = None
        self.data_val: Optional[List[TaskEncoding]] = None

    @property
    def num_train(self) -> int:
        return self.train_val_split[0]

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val."""
        train_documents = load_multiconer(data_dir=self.data_dir, name=self.name, split="train")
        val_documents = load_multiconer(data_dir=self.data_dir, name=self.name, split="validation")

        self.task_module.prepare(train_documents)

        self.data_train = self.task_module.encode(train_documents, encode_target=True)
        self.data_val = self.task_module.encode(val_documents, encode_target=True)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=False,
        )
