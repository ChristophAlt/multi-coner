from typing import List, Optional, Dict, Any

from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_ie.data.datasets.json import load_json

from pytorch_ie.data.document import Document, LabeledSpan

CLASS_NAME_TO_CLASS = {
    "LabeledSpan": LabeledSpan,
}


def load_json(data: Dict[str, Any], text_field: str="text") -> List[Document]:

    annotations = {
        name: CLASS_NAME_TO_CLASS[class_name] for name, class_name in data["annotations"].items()
    }

    documents: List[Document] = []
    for example in data["documents"]:
        text = example.get(text_field)
        assert text is not None, "text_field does not exist."

        document = Document(text)
        for name, cls in annotations.items():
            if name in example:
                field = example[name]
                for dct in field:
                    annotation = cls.from_dict(dct)
                    document.add_annotation(name, annotation)

        documents.append(document)

    return documents


DOCUMENTS = {
    "annotations": {
        "entities": "LabeledSpan",
        "sentences": "LabeledSpan"
    },
    "documents": [
        {
            "text": "Jane lives in Berlin. Berlin is in Germany. This is another sentence.",
            "entities": [
                {"start": 0, "end": 4, "label": "PER"},
                {"start": 14, "end": 20, "label": "LOC"},
                {"start": 22, "end": 28, "label": "LOC"},
                {"start": 35, "end": 42, "label": "LOC"}
            ],
            "sentences": [
                {"start": 0, "end": 21, "label": "1"},
                {"start": 23, "end": 43, "label": "2"},
                {"start": 44, "end": 69, "label": "3"}
            ]
        },
        {
            "text": "John works at Hamboldt University.",
            "entities": [
                {"start": 0, "end": 4, "label": "PER"},
                {"start": 14, "end": 33, "label": "ORG"}
            ],
            "sentences": [
                {"start": 0, "end": 34, "label": "1"}
            ]
        },
        {
            "text": "Jim works at Hamburg University.",
            "entities": [
                {"start": 0, "end": 3, "label": "PER"},
                {"start": 13, "end": 31, "label": "ORG"}
            ],
            "sentences": [
                {"start": 0, "end": 32, "label": "1"}
            ]
        }
    ]
}


class TestDataModule(LightningDataModule):
    def __init__(
        self,
        task_module: TaskModule,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.task_module = task_module
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
        train_documents = load_json(DOCUMENTS)
        val_documents = load_json(DOCUMENTS)

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
