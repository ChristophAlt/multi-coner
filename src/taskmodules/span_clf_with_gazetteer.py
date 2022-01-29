import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from gensim.models import KeyedVectors

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models.transformer_span_classification import (
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule

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


class Gazetteer:
    def __init__(self, path: str, lowercase: bool = True) -> None:
        self.dictionary = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()

                if line:
                    string, label = line.split("\t")[:2]

                    if lowercase:
                        string = string.lower()

                    if string not in self.dictionary:
                        self.dictionary[string] = set()

                    self.dictionary[string].add(label)

        # labels = ["CORP", "PROD", "GRP", "CW", "LOC", "PER"]
        labels = set()
        for val in self.dictionary.values():
            labels.update(val)

        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.num_labels = len(self.label_to_id)

        logger.debug("label_to_id: %s", self.label_to_id)
        logger.debug("num_labels: %d", self.num_labels)

    def lookup(self, text: str):
        return self.dictionary.get(text, set())


class EfficientGazetteer:
    def __init__(self, path: str, tokenizer, lowercase: bool = True) -> None:
        self.dictionary = {}
        with open(path, "r") as f:
            for line in tqdm.tqdm(f.readlines()):
                line = line.strip()

                if line:
                    string, label = line.split("\t")[:2]

                    if lowercase:
                        string = string.lower()

                    ids = tuple(tokenizer(string, add_special_tokens=False)["input_ids"])

                    if ids not in self.dictionary:
                        self.dictionary[ids] = set()

                    self.dictionary[ids].add(label)

        all_labels = set()
        for val in self.dictionary.values():
            all_labels.update(val)

        self.label_to_id = {label: i for i, label in enumerate(all_labels)}
        self.num_labels = len(self.label_to_id)

    def lookup(self, ids: List[int]):
        return self.dictionary.get(ids, set())


class SpanClassificationWithGazetteerTaskModule(_TransformerSpanClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        wiki_to_vec_file: str,
        gazetteer_path: str,
        use_efficient_gazetteer: bool = True,
        entity_annotation: str = "entities",
        single_sentence: bool = False,
        sentence_annotation: str = "sentences",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
        max_span_length: int = 8,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            wiki_to_vec_file=wiki_to_vec_file,
            entity_annotation=entity_annotation,
            single_sentence=single_sentence,
            sentence_annotation=sentence_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            max_span_length=max_span_length,
            gazetteer_path=gazetteer_path,
            use_efficient_gazetteer=use_efficient_gazetteer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.use_efficient_gazetteer = use_efficient_gazetteer

        if self.use_efficient_gazetteer:
            def prepare_key(key):
                key = key.replace("/", " / ").replace("_", " ")
                ids = tuple(self.tokenizer(key, add_special_tokens=False)["input_ids"])
                return ids

            self.wiki_to_vec_key_to_index = {prepare_key(key.lower()): index + 1 for key, index in tqdm.tqdm(KeyedVectors.load(wiki_to_vec_file).key_to_index.items())}
            self.gazetteer = EfficientGazetteer(gazetteer_path, tokenizer=self.tokenizer, lowercase=True)
        else:
            self.wiki_to_vec_key_to_index = {key.lower(): index + 1 for key, index in KeyedVectors.load(wiki_to_vec_file).key_to_index.items()}
            self.gazetteer = Gazetteer(gazetteer_path, lowercase=True)

        self.entity_annotation = entity_annotation
        self.single_sentence = single_sentence
        self.sentence_annotation = sentence_annotation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.max_span_length = max_span_length

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
        config = super()._config()
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

    def _add_wikipedia_entities(self, input_, metadata):
        entity_ids = tuple(self.tokenizer("entity /", add_special_tokens=False)["input_ids"])

        for inp, metad in tqdm.tqdm(zip(input_, metadata)):
            inp_ids = inp["input_ids"]
            seq_length = len(inp_ids)

            entity_spans = []
            gaz_features = []
            metad["position_ids"] = list(range(seq_length))
            metad["seq_len"] = seq_length
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length

                    span_inp_ids = inp_ids[start_index:end_index]

                    if self.use_efficient_gazetteer:
                        entity_index = self.wiki_to_vec_key_to_index.get(entity_ids + tuple(span_inp_ids))
                        gaz_labels = self.gazetteer.lookup(tuple(span_inp_ids))
                    else:
                        span_text = self.tokenizer.decode(span_inp_ids)
                        entity_index = self.wiki_to_vec_key_to_index.get("entity/" + span_text.replace(" ", "_"))
                        gaz_labels = self.gazetteer.lookup(span_text)

                    if entity_index is not None:
                        entity_spans.append((start_index, end_index - 1, span_length, entity_index))

                    if len(span_text) <= 3:
                        continue

                    if end_index < seq_length - 1:
                        next_token = self.tokenizer.convert_ids_to_tokens(inp_ids[end_index])

                        # not at end of word
                        if next_token.startswith("##"):
                            continue

                    if gaz_labels:
                        gaz_feat = [0] * self.gazetteer.num_labels
                        for label in gaz_labels:
                            gaz_feat[self.gazetteer.label_to_id[label]] = 1

                        gaz_features.append((start_index, end_index - 1, span_length, gaz_feat))

                    #     for label in gaz_labels:
                    #         inp["input_ids"].extend([self.special_tokens_to_id[f"[{label}]"], self.special_tokens_to_id[f"[/{label}]"]])
                    #         metad["position_ids"].extend([start_index, end_index - 1])
                    #         inp["token_type_ids"].extend([1, 1])
                    #         inp["attention_mask"].extend([1, 1])
                    #         inp["special_tokens_mask"].extend([1, 1])

            # inp["input_ids"].append(self.tokenizer.vocab["[SEP]"])
            # metad["position_ids"].append(seq_length + 1)
            # inp["token_type_ids"].append(0)
            # inp["attention_mask"].append(1)

            metad["wikipedia_entities"] = entity_spans
            metad["gazetteer"] = gaz_features

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[
        List[TransformerSpanClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        expanded_documents = None
        if self.single_sentence:
            sent: LabeledSpan
            input_ = [
                self.tokenizer(
                    doc.text[sent.start : sent.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in documents
                for sent in doc.span_annotations(self.sentence_annotation)
            ]
            expanded_documents = [
                doc for doc in documents for _ in doc.span_annotations(self.sentence_annotation)
            ]
        else:
            input_ = [
                self.tokenizer(
                    doc.text,
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in tqdm.tqdm(documents)
            ]

        metadata = [
            {
                "offset_mapping": inp.pop("offset_mapping"),
                # "special_tokens_mask": inp.pop("special_tokens_mask"),
            }
            for inp in input_
        ]

        if self.single_sentence:
            i = 0
            for document in documents:
                for sentence_index in range(
                    len(document.span_annotations(self.sentence_annotation))
                ):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        self._add_wikipedia_entities(input_, metadata)

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerSpanClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerSpanClassificationTargetEncoding]:
        target = []
        if self.single_sentence:
            for i, document in enumerate(documents):
                entities = document.span_annotations(self.entity_annotation)
                sentence_idx = metadata[i]["sentence_index"]
                sentence = document.span_annotations(self.sentence_annotation)[sentence_idx]

                label_ids = []
                entity: LabeledSpan
                for entity in entities:
                    if entity.start < sentence.start or entity.end > sentence.end:
                        continue

                    entity_start = entity.start - sentence.start
                    entity_end = entity.end - sentence.start

                    start_idx = input_encodings[i].char_to_token(entity_start)
                    end_idx = input_encodings[i].char_to_token(entity_end - 1)
                    # TODO: remove this is if case
                    if start_idx is None or end_idx is None:
                        logger.warning(
                            f"Entity annotation does not start or end with a token, it will be skipped: {entity}"
                        )
                        continue

                    label_ids.append((start_idx, end_idx, self.label_to_id[entity.label_single]))

                target.append(label_ids)
        else:
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
            start = start - batch_idx * output["seq_length"]
            end = end - batch_idx * output["seq_length"]

            label = self.id_to_label[label_id]
            if label != "O":
                tags[batch_idx].append((label, float(prob[label_id]), (start, end)))
                probabilities[batch_idx].append(prob[label_id])

        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        encoding: TransformerSpanClassificationTaskEncoding,
        output: TransformerSpanClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.single_sentence:
            document = encoding.document
            metadata = encoding.metadata

            sentence = document.span_annotations(self.sentence_annotation)[
                metadata["sentence_index"]
            ]

            # tag_sequence = [
            #     "O" if stm else tag
            #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            # ]

            # spans = bio_tags_to_spans(tag_sequence)
            spans = output["tags"]
            for label, prob, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        start=sentence.start + metadata["offset_mapping"][start][0],
                        end=sentence.start + metadata["offset_mapping"][end][1],
                        label=label,
                        score=prob,
                    ),
                )
        else:

            metadata = encoding.metadata

            # tag_sequence = [
            #     "O" if stm else tag
            #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            # ]

            # spans = bio_tags_to_spans(tag_sequence)
            spans = output["tags"]
            for label, prob, (start, end) in spans:
                start = min(start, len(metadata["offset_mapping"]) - 2)
                end = min(end, len(metadata["offset_mapping"]) - 2)
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        start=metadata["offset_mapping"][start][0],
                        end=metadata["offset_mapping"][end][1],
                        label=label,
                        score=prob,
                    ),
                )

    def collate(
        self, encodings: List[TransformerSpanClassificationTaskEncoding]
    ) -> TransformerSpanClassificationModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]
        metadata = [encoding.metadata for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}
        input_ = dict(input_)
        # seq_length = input_["input_ids"].shape[1]

        # position_ids = []
        # for metad in metadata:
        #     start_position = metad["seq_len"]
        #     end_position = start_position + (seq_length - len(metad["position_ids"]))
        #     position_ids.append(list(metad["position_ids"]) + list(range(start_position, end_position)))

        # input_["position_ids"] = torch.tensor(position_ids, dtype=torch.int64)

        input_["wikipedia_entities"] = [metad["wikipedia_entities"] for metad in metadata]
        input_["gazetteer"] = [metad["gazetteer"] for metad in metadata]

        if not encodings[0].has_target:
            return input_, None

        target_list: List[TransformerSpanClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]

        return input_, target_list
