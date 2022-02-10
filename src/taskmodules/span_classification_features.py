import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from gensim.models import KeyedVectors
from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models.transformer_span_classification import (
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from src.gazetteer import Gazetteer

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

TransformerSpanClassificationInputEncoding = BatchEncoding
TransformerSpanClassificationTargetEncoding = List[Tuple[int, int, int]]

TransformerSpanClassificationTaskEncoding = TaskEncoding[
    TransformerSpanClassificationInputEncoding, TransformerSpanClassificationTargetEncoding
]
TransformerSpanClassificationTaskOutput = Dict[str, Any]

_TransformerSpanClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerSpanClassificationInputEncoding,
    TransformerSpanClassificationTargetEncoding,
    TransformerSpanClassificationModelStepBatchEncoding,
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationTaskOutput,
]


logger = logging.getLogger(__name__)


class SpanClassificationWithFeaturesTaskModule(_TransformerSpanClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
        ##
        wiki_to_vec_file: Optional[str] = None,
        gazetteer_path: Optional[str] = None,
        gazetteer_add_input_tokens: bool = False,
        gazetteer_add_output_features: bool = False,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            entity_annotation=entity_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            wiki_to_vec_file=wiki_to_vec_file,
            gazetteer_path=gazetteer_path,
            gazetteer_add_input_tokens=gazetteer_add_input_tokens,
            gazetteer_add_output_features=gazetteer_add_output_features,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.gazetteer_add_input_tokens = gazetteer_add_input_tokens
        self.gazetteer_add_output_features = gazetteer_add_output_features

        self.has_wiki_to_vec = False
        self.has_gazetteer = False

        if wiki_to_vec_file is not None:
            self.has_wiki_to_vec = True

            self.wiki_to_vec_key_to_index = {
                key.lower(): index + 1
                for key, index in KeyedVectors.load(wiki_to_vec_file).key_to_index.items()
            }

        if gazetteer_path is not None:
            self.has_gazetteer = True
            self.gazetteer = Gazetteer(gazetteer_path, lowercase=True)

            if self.gazetteer_add_input_tokens:
                special_tokens = []
                for label in self.gazetteer.label_to_id.keys():
                    special_tokens.append(f"[{label}]")
                    special_tokens.append(f"[/{label}]")

                self.tokenizer.add_tokens(special_tokens, special_tokens=True)

                self.special_tokens_to_id = {}
                for label in self.gazetteer.label_to_id.keys():
                    self.special_tokens_to_id[f"[{label}]"] = self.tokenizer.vocab[f"[{label}]"]
                    self.special_tokens_to_id[f"[/{label}]"] = self.tokenizer.vocab[f"[/{label}]"]

    def _config(self) -> Dict[str, Any]:
        config = dict(super()._config())
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.span_annotations(self.entity_annotation)

            for entity in entities:
                # TODO: labels is a set, use update
                for label in entity.labels:
                    if label not in labels:
                        labels.add(label)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in labels:
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _add_features(self, inputs, metadata):
        for inp, metad in tqdm.tqdm(zip(inputs, metadata)):
            inp_ids = inp["input_ids"]
            seq_length = len(inp_ids)

            metad["position_ids"] = list(range(seq_length))
            metad["seq_len"] = seq_length

            entity_spans = []
            gaz_features = []
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length

                    span_inp_ids = inp_ids[start_index:end_index]

                    span_text = self.tokenizer.decode(span_inp_ids)

                    if self.has_wiki_to_vec:
                        entity_index = self.wiki_to_vec_key_to_index.get(
                            "entity/" + span_text.replace(" ", "_")
                        )

                        if entity_index is not None:
                            entity_spans.append(
                                (start_index, end_index - 1, span_length, entity_index)
                            )

                    if self.has_gazetteer:
                        if len(span_text) <= 3:
                            continue

                        if end_index < seq_length - 1:
                            next_token = self.tokenizer.convert_ids_to_tokens(inp_ids[end_index])

                            # not at end of word
                            if next_token.startswith("##"):
                                continue

                        gaz_labels = self.gazetteer.lookup(span_text)

                        if gaz_labels:
                            if self.gazetteer_add_output_features:
                                gaz_feat = [0] * self.gazetteer.num_labels
                                for label in gaz_labels:
                                    gaz_feat[self.gazetteer.label_to_id[label]] = 1

                                gaz_features.append(
                                    (start_index, end_index - 1, span_length, gaz_feat)
                                )

                            if self.gazetteer_add_input_tokens:
                                for label in gaz_labels:
                                    inp["input_ids"].extend(
                                        [
                                            self.special_tokens_to_id[f"[{label}]"],
                                            self.special_tokens_to_id[f"[/{label}]"],
                                        ]
                                    )
                                    metad["position_ids"].extend([start_index, end_index - 1])
                                    inp["token_type_ids"].extend([0, 0])
                                    inp["attention_mask"].extend([1, 1])
                                    inp["special_tokens_mask"].extend([1, 1])

            if self.has_wiki_to_vec:
                metad["wikipedia_entities"] = entity_spans

            if self.has_gazetteer and self.gazetteer_add_output_features:
                metad["gazetteer"] = gaz_features

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[
        List[TransformerSpanClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        inputs = [
            self.tokenizer(
                doc.text,
                padding=False,
                truncation=self.truncation,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            for doc in documents
        ]

        metadata = [
            {
                "offset_mapping": inp.pop("offset_mapping"),
                # "special_tokens_mask": inp.pop("special_tokens_mask"),
            }
            for inp in inputs
        ]

        if self.has_wiki_to_vec or self.has_gazetteer:
            self._add_features(inputs, metadata)

        return inputs, metadata, None

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerSpanClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerSpanClassificationTargetEncoding]:
        target = []
        for i, document in enumerate(documents):
            entities = document.span_annotations(self.entity_annotation)

            label_ids = []
            for entity in entities:
                start_idx = input_encodings[i].char_to_token(entity.start)
                end_idx = input_encodings[i].char_to_token(entity.end - 1)
                label_ids.append((start_idx, end_idx, self.label_to_id[entity.label_single]))

            target.append(label_ids)

        return target

    def unbatch_output(
        self, output: TransformerSpanClassificationModelBatchOutput
    ) -> Sequence[TransformerSpanClassificationTaskOutput]:
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        start_indices = output["start_indices"].detach().cpu().numpy()
        end_indices = output["end_indices"].detach().cpu().numpy()
        batch_indices = output["batch_indices"].detach().cpu().numpy()

        tags: List[List[Tuple[str, Tuple[int, int]]]] = [[] for _ in np.unique(batch_indices)]
        probabilities: List[List[float]] = [[] for _ in np.unique(batch_indices)]
        for start, end, batch_idx, label_id, prob in zip(
            start_indices, end_indices, batch_indices, label_ids, probs
        ):
            label = self.id_to_label[label_id]
            if label != "O":
                tags[batch_idx].append((label, (start, end)))
                probabilities[batch_idx].append(prob[label_id])

        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        encoding: TransformerSpanClassificationTaskEncoding,
        output: TransformerSpanClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:

        metadata = encoding.metadata

        spans = output["tags"]
        for label, (start, end) in spans:
            yield (
                self.entity_annotation,
                LabeledSpan(
                    metadata["offset_mapping"][start][0],
                    metadata["offset_mapping"][end][1],
                    label,
                ),
            )

    def collate(
        self, encodings: List[TransformerSpanClassificationTaskEncoding]
    ) -> TransformerSpanClassificationModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]
        metadata = [encoding.metadata for encoding in encodings]

        inputs = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        inputs = dict(inputs)

        if self.has_gazetteer and self.gazetteer_add_input_tokens:
            seq_length = inputs["input_ids"].shape[1]

            position_ids = []
            for metad in metadata:
                start_position = metad["seq_len"]
                end_position = start_position + (seq_length - len(metad["position_ids"]))
                position_ids.append(
                    list(metad["position_ids"]) + list(range(start_position, end_position))
                )

            inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.int64)

        if self.has_gazetteer and self.gazetteer_add_output_features:
            inputs["gazetteer"] = [metad["gazetteer"] for metad in metadata]

        if self.has_wiki_to_vec:
            inputs["wikipedia_entities"] = [metad["wikipedia_entities"] for metad in metadata]

        if not encodings[0].has_target:
            return inputs, None

        targets: List[TransformerSpanClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]

        return inputs, targets
