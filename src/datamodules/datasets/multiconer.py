from typing import List, Union

import os
from src.datamodules.datasets import HF_DATASETS_ROOT
from datasets import Dataset, IterableDataset, load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, LabeledSpan
from pytorch_ie.data.span_utils import bio_tags_to_spans


def load_multiconer(
    data_dir: str,
    name: str,
    split: Union[str, Split],
) -> List[Document]:
    path = "multiconer"
    data = load_dataset(
        path=os.path.join(HF_DATASETS_ROOT, "multiconer"),
        name=name,
        data_dir=data_dir,
        split=split,
    )
    assert isinstance(data, (Dataset, IterableDataset)), (
        f"`load_dataset` for `path={path}` and `split={split}` should return a single Dataset, "
        f"but the result is of type: {type(data)}"
    )

    int_to_str = data.features["ner_tags"].feature.int2str

    documents = []
    for example in data:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        start = 0
        token_offsets = []
        tag_sequence = []
        for token, tag_id in zip(tokens, ner_tags):
            end = start + len(token)
            token_offsets.append((start, end))
            tag_sequence.append(int_to_str(tag_id))

            start = end + 1

        text = " ".join(tokens)
        spans = bio_tags_to_spans(tag_sequence)

        document = Document(text, doc_id=example["id"])

        for label, (start, end) in spans:
            start_offset = token_offsets[start][0]
            end_offset = token_offsets[end][1]
            document.add_annotation(
                "entities", LabeledSpan(start=start_offset, end=end_offset, label=label)
            )

        documents.append(document)

    return documents
