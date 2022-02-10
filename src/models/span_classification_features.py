from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchmetrics
from gensim.models import KeyedVectors
from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.models.modules.mlp import MLP
from torch import Tensor, nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)

TransformerSpanClassificationModelBatchEncoding = BatchEncoding
TransformerSpanClassificationModelBatchOutput = Dict[str, Any]

TransformerSpanClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[List[List[Tuple[int, int, int]]]],
]


class SpanClassificationWithFeaturesModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        t_total: int,
        tokenizer_vocab_size: int,
        tokenizer_mask_token_id: int = 100,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-5,
        warmup_proportion: float = 0.1,
        warmup_schedule: str = "none",
        ignore_index: int = 0,
        max_span_length: int = 8,
        span_length_embedding_dim: int = 150,
        freeze_model: bool = False,
        layer_mean: bool = False,
        augment_input: bool = False,
        augment_input_prob: float = 0.2,
        use_mlp: bool = True,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        dropout_prob: float = 0.1,
        ##
        use_gazetteer: bool = False,
        use_wiki_to_vec: bool = False,
        gazetteer_add_input_tokens: bool = False,
        gazetteer_add_output_features: bool = False,
        wiki_to_vec_file: Optional[str] = None,
        num_gazetteer_labels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if use_gazetteer and gazetteer_add_output_features and num_gazetteer_labels is None:
            raise ValueError(
                "'use_gazetteer' in combination with 'gazetteer_add_output_features' requires 'num_gazetteer_labels' to be set."
            )

        if use_wiki_to_vec and wiki_to_vec_file is None:
            raise ValueError("'use_wiki_to_vec' requires 'wiki_to_vec_file' to be set.")

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.warmup_schedule = warmup_schedule
        self.max_span_length = max_span_length
        self.layer_mean = layer_mean
        self.augment_input = augment_input
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.tokenizer_mask_token_id = tokenizer_mask_token_id
        self.augment_input_prob = augment_input_prob
        self.dropout_prob = dropout_prob

        self.use_gazetteer = use_gazetteer
        self.use_wiki_to_vec = use_wiki_to_vec
        self.gazetteer_add_input_tokens = gazetteer_add_input_tokens
        self.gazetteer_add_output_features = gazetteer_add_output_features
        self.num_gazetteer_labels = num_gazetteer_labels

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(self.tokenizer_vocab_size)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        joint_embedding_dim = config.hidden_size * 2 + span_length_embedding_dim

        if self.use_wiki_to_vec:
            entity_embeddings = torch.from_numpy(KeyedVectors.load(wiki_to_vec_file).vectors)
            entity_embedding_dim = entity_embeddings.shape[1]
            embedding_weights = torch.cat(
                [torch.zeros(1, entity_embedding_dim), entity_embeddings], axis=0
            )
            self.entity_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

            joint_embedding_dim += entity_embedding_dim

        if self.use_gazetteer and self.gazetteer_add_output_features:
            assert self.num_gazetteer_labels is not None
            joint_embedding_dim += self.num_gazetteer_labels

        self.span_length_embedding = nn.Embedding(
            num_embeddings=max_span_length, embedding_dim=span_length_embedding_dim
        )

        self.layer_norm = nn.LayerNorm(joint_embedding_dim)

        self.dropout = nn.Dropout(self.dropout_prob)

        if use_mlp:
            self.classifier = MLP(
                input_dim=joint_embedding_dim,
                output_dim=num_classes,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_num_layers,
            )
        else:
            self.classifier = nn.Linear(
                in_features=joint_embedding_dim,
                out_features=num_classes,
            )

        self.loss_fct = nn.CrossEntropyLoss()

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def _start_end_and_span_length_span_index(
        self, batch_size: int, max_seq_length: int, seq_lengths: Optional[Iterable[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        start_indices = []
        end_indices = []
        span_lengths = []
        span_batch_index = []
        offsets = []
        for batch_index, seq_length in enumerate(seq_lengths):
            offset = max_seq_length * batch_index

            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    span_batch_index.append(batch_index)
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    span_lengths.append(span_length - 1)
                    offsets.append(offset)

        return (
            torch.tensor(start_indices),
            torch.tensor(end_indices),
            torch.tensor(span_lengths),
            torch.tensor(span_batch_index),
            torch.tensor(offsets),
        )

    def _expand_target_tuples(
        self,
        target_tuples: List[List[Tuple[int, int, int]]],
        batch_size: int,
        max_seq_length: int,
        seq_lengths: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        target = []
        for batch_index, seq_length in enumerate(seq_lengths):
            label_lookup = {
                (start, end): label for start, end, label in target_tuples[batch_index]
            }
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    label = label_lookup.get((start_index, end_index), 0)
                    target.append(label)

        return torch.tensor(target)

    def _mask_input_tokens(self, inputs: Any) -> Tuple[Any, Any]:
        input_ids = inputs["input_ids"].clone()

        special_tokens_mask = inputs["special_tokens_mask"]
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix = torch.full(
            input_ids.shape, self.augment_input_prob, device=input_ids.device
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer_mask_token_id
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer_mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(self.tokenizer_vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
        # input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids

    def _get_additional_embeddings(
        self,
        wikipedia_entities,
        gazetteer_features,
        batch_size: int,
        device,
        max_seq_length: int,
        seq_lengths: Optional[Iterable[int]] = None,
    ) -> List[torch.Tensor]:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        add_wiki_features = self.use_wiki_to_vec
        add_gazetteer_features = self.use_gazetteer and self.gazetteer_add_output_features

        # all_zeros = [0] * self.num_gazetteer_labels

        wiki_indices = []
        gaz_features = []
        for batch_index, seq_length in enumerate(seq_lengths):
            self.wikipedia_lookup = None
            if add_wiki_features:
                assert wikipedia_entities is not None
                wikipedia_lookup = {
                    (start, end, span_length): label
                    for start, end, span_length, label in wikipedia_entities[batch_index]
                }

            self.gazetteer_lookup = None
            if add_gazetteer_features:
                assert gazetteer_features is not None
                gazetteer_lookup = {
                    (start, end, span_length): features
                    for start, end, span_length, features in gazetteer_features[batch_index]
                }

            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    if add_wiki_features:
                        wiki_index = wikipedia_lookup.get((start_index, end_index, span_length), 0)
                        wiki_indices.append(wiki_index)

                    if add_gazetteer_features:
                        gaz_feat = gazetteer_lookup.get(
                            (start_index, end_index, span_length), [0] * self.num_gazetteer_labels
                        )
                        gaz_features.append(gaz_feat)

        additional_embeddings = []

        if add_wiki_features:
            wiki_ids = torch.tensor(wiki_indices).to(device)
            entity_embeddings = self.entity_embeddings(wiki_ids)
            additional_embeddings.append(entity_embeddings)

        if add_gazetteer_features:
            gazetter_feature_embeddings = torch.tensor(gaz_features).to(device)
            additional_embeddings.append(gazetter_feature_embeddings)

        return additional_embeddings

    def forward(self, inputs: TransformerSpanClassificationModelBatchEncoding) -> TransformerSpanClassificationModelBatchOutput:  # type: ignore
        inputs.pop("special_tokens_mask", None)

        wikipedia_entities = inputs.pop("wikipedia_entities", None)
        gazetteer_features = inputs.pop("gazetteer", None)

        output = self.model(**inputs, output_hidden_states=self.layer_mean)

        batch_size, seq_length, hidden_dim = output.last_hidden_state.shape

        if self.layer_mean:
            hidden_state = torch.mean(torch.stack(output.hidden_states), dim=0).view(
                batch_size * seq_length, hidden_dim
            )
        else:
            hidden_state = output.last_hidden_state.view(batch_size * seq_length, hidden_dim)

        seq_lengths = None
        if "attention_mask" in inputs:
            seq_lengths = torch.sum(inputs["attention_mask"], dim=-1).detach().cpu()

        (
            start_indices,
            end_indices,
            span_length,
            batch_indices,
            offsets,
        ) = self._start_end_and_span_length_span_index(
            batch_size=batch_size, max_seq_length=seq_length, seq_lengths=seq_lengths
        )

        start_embedding = hidden_state[offsets + start_indices, :]
        end_embedding = hidden_state[offsets + end_indices, :]
        span_length_embedding = self.span_length_embedding(span_length.to(hidden_state.device))

        all_embeddings = [start_embedding, end_embedding, span_length_embedding]
        all_embeddings.extend(
                self._get_additional_embeddings(
                wikipedia_entities,
                gazetteer_features,
                batch_size=batch_size,
                device=hidden_state.device,
                max_seq_length=seq_length,
                seq_lengths=seq_lengths,
            )
        )

        joint_embedding = torch.cat(all_embeddings, dim=-1)

        logits = self.classifier(self.dropout(self.layer_norm(joint_embedding)))

        return {
            "logits": logits,
            "batch_indices": batch_indices,
            "start_indices": start_indices,
            "end_indices": end_indices,
        }

    def _shared_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, augment_inputs: bool):  # type: ignore
        inputs, target_tuples = batch
        assert target_tuples is not None, "targets must be available for training"

        batch_size, seq_length = inputs["input_ids"].shape

        seq_length_without_padding = None
        if "attention_mask" in inputs:
            seq_length_without_padding = torch.sum(inputs["attention_mask"], dim=-1)

        if augment_inputs:
            inputs["input_ids"] = self._mask_input_tokens(inputs)

        output = self(inputs)

        logits = output["logits"]

        targets = self._expand_target_tuples(
            target_tuples=target_tuples,
            batch_size=batch_size,
            max_seq_length=seq_length,
            seq_lengths=seq_length_without_padding,
        )
        targets = targets.to(logits.device)

        loss = self.loss_fct(logits, targets)

        return loss, logits, targets

    def training_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        loss, logits, targets = self._shared_step(batch, augment_inputs=self.augment_input)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_f1(logits, targets)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        loss, logits, targets = self._shared_step(batch, augment_inputs=False)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_f1(logits, targets)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        param_optimizer = [
            (name, param) for name, param in self.named_parameters() if param.requires_grad
        ]

        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in param_optimizer
                    if "classifier" not in name and "span_length_embedding" not in name
                ]
            },
            {
                "params": [
                    param
                    for name, param in param_optimizer
                    if "classifier" in name or "span_length_embedding" in name
                ],
                "lr": self.task_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        return [optimizer]
