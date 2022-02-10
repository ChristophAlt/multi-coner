from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchmetrics
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


class SpanClassificationModel(PyTorchIEModel):
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
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(self.tokenizer_vocab_size)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        joint_embedding_dim = self._get_joint_embedding_dim(
            model_hidden_dim=config.hidden_size,
            span_length_embedding_dim=span_length_embedding_dim,
        )

        self.span_length_embedding = nn.Embedding(
            num_embeddings=max_span_length, embedding_dim=span_length_embedding_dim
        )

        self.layer_norm = nn.LayerNorm(joint_embedding_dim)

        self.dropout = nn.Dropout(self.dropout_prob)

        self.classifier = MLP(
            input_dim=joint_embedding_dim,
            output_dim=num_classes,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def _get_joint_embedding_dim(
        self, model_hidden_dim: int, span_length_embedding_dim: int
    ) -> int:
        return model_hidden_dim * 2 + span_length_embedding_dim

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
        self, inputs: TransformerSpanClassificationModelBatchEncoding
    ) -> List[torch.Tensor]:
        return []

    def forward(self, inputs: TransformerSpanClassificationModelBatchEncoding) -> TransformerSpanClassificationModelBatchOutput:  # type: ignore
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

        joint_embedding = torch.cat(
            (start_embedding, end_embedding, span_length_embedding), dim=-1
        )

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
            (param, name) for param, name in self.named_parameters() if param.requires_grad
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