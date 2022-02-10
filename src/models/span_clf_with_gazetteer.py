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

from src.taskmodules.span_clf_with_gazetteer import Gazetteer

TransformerSpanClassificationModelBatchEncoding = BatchEncoding
TransformerSpanClassificationModelBatchOutput = Dict[str, Any]

TransformerSpanClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[List[List[Tuple[int, int, int]]]],
]


class SpanClassificationWithGazetteerModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        t_total: int,
        wiki_to_vec_file: str,
        num_gazetteer_labels: int,
        tokenizer_vocab_size: int,
        tokenizer_unk_token_id: int = 100,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-5,
        warmup_proportion: float = 0.1,
        ignore_index: int = 0,
        max_span_length: int = 8,
        span_length_embedding_dim: int = 150,
        freeze_model: bool = False,
        layer_mean: bool = False,
        augment_input: bool = False,
        augment_mask_probability: float = 0.25,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.max_span_length = max_span_length
        self.layer_mean = layer_mean
        self.augment_input = augment_input
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.tokenizer_unk_token_id = tokenizer_unk_token_id
        self.augment_mask_probability = augment_mask_probability

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(self.tokenizer_vocab_size)

        entity_embeddings = torch.from_numpy(KeyedVectors.load(wiki_to_vec_file).vectors)
        entity_embedding_dim = entity_embeddings.shape[1]
        embedding_weights = torch.cat(
            [torch.zeros(1, entity_embedding_dim), entity_embeddings], axis=0
        )
        self.entity_embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.dropout = nn.Dropout(0.5)
        # TODO: properly intialize!
        # self.classifier = nn.Linear(config.hidden_size * 2 + span_length_embedding_dim, num_classes)

        self.num_gazetteer_labels = num_gazetteer_labels

        self.classifier = MLP(
            input_dim=config.hidden_size * 2
            + span_length_embedding_dim
            + entity_embedding_dim
            + self.num_gazetteer_labels,
            output_dim=num_classes,
            hidden_dim=1024,
            num_layers=2,
        )

        self.span_length_embedding = nn.Embedding(
            num_embeddings=max_span_length, embedding_dim=span_length_embedding_dim
        )

        self.layer_norm = nn.LayerNorm(
            config.hidden_size * 2
            + span_length_embedding_dim
            + entity_embedding_dim
            + self.num_gazetteer_labels
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def _start_end_and_span_length_span_index(
        self,
        batch_size: int,
        max_seq_length: int,
        seq_lengths: Optional[Iterable[int]] = None,
        wikipedia_entities=None,
        gazetteer_features=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        all_zeros = [0] * self.num_gazetteer_labels

        start_indices = []
        end_indices = []
        span_lengths = []
        span_batch_index = []
        wiki_indices = []
        gaz_features = []
        for batch_index, seq_length in enumerate(seq_lengths):
            wikipedia_lookup = {
                (start, end, span_length): label
                for start, end, span_length, label in wikipedia_entities[batch_index]
            }
            gazetteer_lookup = {
                (start, end, span_length): features
                for start, end, span_length, features in gazetteer_features[batch_index]
            }
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    offset = max_seq_length * batch_index

                    span_batch_index.append(batch_index)
                    start_indices.append(offset + start_index)
                    end_indices.append(offset + end_index)
                    span_lengths.append(span_length - 1)
                    wiki_index = wikipedia_lookup.get((start_index, end_index, span_length), 0)
                    wiki_indices.append(wiki_index)
                    gaz_feat = gazetteer_lookup.get(
                        (start_index, end_index, span_length), all_zeros
                    )
                    gaz_features.append(gaz_feat)

        return (
            torch.tensor(start_indices),
            torch.tensor(end_indices),
            torch.tensor(span_lengths),
            torch.tensor(span_batch_index),
            torch.tensor(wiki_indices),
            torch.tensor(gaz_features),
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

    def _torch_mask_tokens(self, inputs: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        input_ids = inputs["input_ids"].clone()

        special_tokens_mask = inputs["special_tokens_mask"]
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix = torch.full(
            input_ids.shape, self.augment_mask_probability, device=input_ids.device
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.unk_token ([UNK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer_unk_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(self.tokenizer_vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
        # input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids

    def forward(self, input_: TransformerSpanClassificationModelBatchEncoding) -> TransformerSpanClassificationModelBatchOutput:  # type: ignore
        input_.pop("special_tokens_mask", None)
        wikipedia_entities = input_.pop("wikipedia_entities", None)
        gazetteer_features = input_.pop("gazetteer", None)
        output = self.model(**input_, output_hidden_states=True)

        batch_size, seq_length, hidden_dim = output.last_hidden_state.shape

        if self.layer_mean:
            hidden_state = torch.mean(torch.stack(output.hidden_states), dim=0).view(
                batch_size * seq_length, hidden_dim
            )
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
            wiki_indices,
            gaz_features,
        ) = self._start_end_and_span_length_span_index(
            batch_size=batch_size,
            max_seq_length=seq_length,
            seq_lengths=seq_lengths,
            wikipedia_entities=wikipedia_entities,
            gazetteer_features=gazetteer_features,
        )

        start_embedding = hidden_state[start_indices, :]
        end_embedding = hidden_state[end_indices, :]
        span_length_embedding = self.span_length_embedding(span_length.to(hidden_state.device))
        wiki_embedding = self.entity_embeddings(wiki_indices.to(hidden_state.device))
        gazetteer_embedding = gaz_features.to(hidden_state.device)

        combined_embedding = torch.cat(
            (
                start_embedding,
                end_embedding,
                span_length_embedding,
                wiki_embedding,
                gazetteer_embedding,
            ),
            dim=-1
            # (start_embedding, end_embedding, span_length_embedding, wiki_embedding), dim=-1
        )

        logits = self.classifier(self.dropout(self.layer_norm(combined_embedding)))

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

        if self.augment_input:
            input_["input_ids"] = self._torch_mask_tokens(input_)

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
        param_optimizer = list(self.named_parameters())
        # param_optimizer = filter(lambda p: p.requires_grad, self.parameters())

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if "classifier" not in n and "span_length_embedding" not in n
                ]
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if "classifier" in n or "span_length_embedding" in n
                ],
                "lr": self.task_learning_rate,
            },
        ]
        # print("Parameters with learning_rate: ", [n for n, p in param_optimizer if "classifier" not in n])
        # print("Parameters with task_learning_rate: ", [n for n, p in param_optimizer if "classifier" in n or "span_length_embedding" in n])
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        # print([n for n, p in self.named_parameters() if p.requires_grad])
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        # print("Optim, Learning rates: ", [group["lr"] for group in optimizer.param_groups])
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        # )
        # return optimizer
        return [optimizer]  # , [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
