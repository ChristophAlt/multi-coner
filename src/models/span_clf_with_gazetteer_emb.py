from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import torch
import torchmetrics
from torch import Tensor, nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)

from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.models.modules.mlp import MLP
from src.models.split_embedding import SplitEmbedding

TransformerSpanClassificationModelBatchEncoding = BatchEncoding
TransformerSpanClassificationModelBatchOutput = Dict[str, Any]

TransformerSpanClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[List[List[Tuple[int, int, int]]]],
]


logger = logging.getLogger(__name__)


class SpanClassificationWithGazetteerEmbeddingModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        t_total: int,
        tokenizer_vocab_size: int,
        additional_embeddings_path: str,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-5,
        warmup_proportion: float = 0.1,
        ignore_index: int = 0,
        max_span_length: int = 8,
        span_length_embedding_dim: int = 150,
        freeze_model: bool = False,
        layer_mean: bool = False,
        threshold_value: int = 100000,
        use_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.max_span_length = max_span_length
        self.layer_mean = layer_mean
        self.threshold_value = threshold_value

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(tokenizer_vocab_size)

        additional_embeddings = nn.Embedding.from_pretrained(torch.load(additional_embeddings_path), freeze=True)

        split_embedding = SplitEmbedding(
            input_embeddings=self.model.get_input_embeddings(),
            additional_embeddings=additional_embeddings,
            threshold_value=self.threshold_value,
        )

        self.model.set_input_embeddings(split_embedding)

        # entity_embeddings = torch.from_numpy(KeyedVectors.load(wiki_to_vec_file).vectors)
        # entity_embedding_dim = entity_embeddings.shape[1]
        # embedding_weights = torch.cat([torch.zeros(1, entity_embedding_dim), entity_embeddings], axis=0)
        # self.entity_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if use_mlp:
            self.classifier = MLP(
                input_dim=config.hidden_size * 2 + span_length_embedding_dim,
                output_dim=num_classes,
                hidden_dim=150,
                num_layers=2,
            )
        else:
            # TODO: properly intialize!
            self.classifier = nn.Linear(config.hidden_size * 2 + span_length_embedding_dim, num_classes)

        self.span_length_embedding = nn.Embedding(
            num_embeddings=max_span_length, embedding_dim=span_length_embedding_dim
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def _start_end_and_span_length_span_index(
        self, batch_size: int, max_seq_length: int, seq_lengths: Optional[Iterable[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        start_indices = []
        end_indices = []
        span_lengths = []
        span_batch_index = []
        for batch_index, seq_length in enumerate(seq_lengths):
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    offset = max_seq_length * batch_index

                    span_batch_index.append(batch_index)
                    start_indices.append(offset + start_index)
                    end_indices.append(offset + end_index)
                    span_lengths.append(span_length - 1)

        return (
            torch.tensor(start_indices),
            torch.tensor(end_indices),
            torch.tensor(span_lengths),
            torch.tensor(span_batch_index),
        )

    # TODO: this should live in the taskmodule
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

    def forward(self, input_: TransformerSpanClassificationModelBatchEncoding) -> TransformerSpanClassificationModelBatchOutput:  # type: ignore
        output = self.model(**input_, output_hidden_states=True)

        batch_size, seq_length, hidden_dim = output.last_hidden_state.shape

        if self.layer_mean:
            hidden_state = torch.mean(torch.stack(output.hidden_states), dim=0).view(batch_size * seq_length, hidden_dim)
        else:
            hidden_state = output.last_hidden_state.view(batch_size * seq_length, hidden_dim)

        seq_lengths = None
        if "attention_mask" in input_:
            seq_lengths = torch.sum(input_["attention_mask"], dim=-1).detach().cpu()

        (
            start_indices,
            end_indices,
            span_length,
            batch_indices,
        ) = self._start_end_and_span_length_span_index(
            batch_size=batch_size, max_seq_length=seq_length, seq_lengths=seq_lengths
        )

        start_embedding = hidden_state[start_indices, :]
        end_embedding = hidden_state[end_indices, :]
        span_length_embedding = self.span_length_embedding(span_length.to(hidden_state.device))

        combined_embedding = torch.cat(
            (start_embedding, end_embedding, span_length_embedding), dim=-1
        )

        logits = self.classifier(self.dropout(combined_embedding))

        return {
            "logits": logits,
            "batch_indices": batch_indices,
            "start_indices": start_indices,
            "end_indices": end_indices,
            "seq_length": seq_length,
        }

    def training_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        input_, target_tuples = batch
        assert target_tuples is not None, "target has to be available for training"

        output = self(input_)

        logits = output["logits"]

        batch_size, seq_length = input_["input_ids"].shape
        seq_lengths = None
        if "attention_mask" in input_:
            seq_lengths = torch.sum(input_["attention_mask"], dim=-1)
        # TODO: Why is this not happening in TransformerSpanClassificationTaskModule.collate?
        target = self._expand_target_tuples(
            target_tuples=target_tuples,
            batch_size=batch_size,
            max_seq_length=seq_length,
            seq_lengths=seq_lengths,
        )
        target = target.to(logits.device)

        loss = self.loss_fct(logits, target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_f1(logits, target)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        input_, target_tuples = batch
        assert target_tuples is not None, "target has to be available for validation"

        output = self(input_)

        logits = output["logits"]

        batch_size, seq_length = input_["input_ids"].shape
        seq_lengths = None
        if "attention_mask" in input_:
            seq_lengths = torch.sum(input_["attention_mask"], dim=-1)

        # TODO: Why is this not happening in TransformerSpanClassificationTaskModule.collate?
        target = self._expand_target_tuples(
            target_tuples=target_tuples,
            batch_size=batch_size,
            max_seq_length=seq_length,
            seq_lengths=seq_lengths,
        )
        target = target.to(logits.device)

        loss = self.loss_fct(logits, target)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_f1(logits, target)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # param_optimizer = list(self.named_parameters())

        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in param_optimizer if "classifier" not in n and "span_length_embedding" not in n]},
        #     {
        #         "params": [p for n, p in param_optimizer if "classifier" in n or "span_length_embedding" in n],
        #         "lr": self.task_learning_rate,
        #     },
        # ]
        # print("Parameters with learning_rate: ", [n for n, p in param_optimizer if "classifier" not in n])
        # print("Parameters with task_learning_rate: ", [n for n, p in param_optimizer if "classifier" in n or "span_length_embedding" in n])
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        # print([n for n, p in self.named_parameters() if p.requires_grad])
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        # print("Optim, Learning rates: ", [group["lr"] for group in optimizer.param_groups])
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        # )
        # return optimizer
        return [optimizer]#, [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
