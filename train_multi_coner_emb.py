import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from src.datamodules.datasets.multiconer import load_multiconer
from src.models.span_clf_with_gazetteer_emb import SpanClassificationWithGazetteerEmbeddingModel
from src.taskmodules.span_clf_with_gazetteer_emb import (
    SpanClassificationWithGazetteerEmbeddingTaskModule,
)


class FreezeUnfreezeCallback(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=5):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # print("Learning rates: ", [group["lr"] for group in optimizer.param_groups])
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model,
                optimizer=optimizer,
                # lr=1e-4,
                initial_denom_lr=1.0,
                train_bn=False,
            )

            # for group in optimizer.param_groups:
            #     group["lr"] = 1e-5


def main():
    pl.seed_everything(42)

    additional_embeddings_path = (
        "/home/christoph/Projects/research/multi_coner/data/embeddings/wikipedia_lowercased.pt"
    )
    alias_to_wikidata_id_path = "/home/christoph/Downloads/wikidata5m_alias/wikidata5m_entity.txt"
    wikidata_id_to_index_path = "/home/christoph/Projects/research/multi_coner/data/embeddings/wikipedia_ids_lowercased.txt"

    model_output_path = "/media/christoph/HDD/models/multiconer-en-spanclf-emb/"

    # model_name = "bert-base-uncased"
    tokenizer_name_or_path = "google/electra-large-discriminator"
    model_name_or_path = tokenizer_name_or_path
    # num_epochs = 100
    # batch_size = 16
    num_epochs = 10
    batch_size = 16

    data_dir = "/home/christoph/Downloads/training_data/"

    train_docs = load_multiconer(
        data_dir=data_dir,
        name="en",
        split="train",
    )

    val_docs = load_multiconer(
        data_dir=data_dir,
        name="en",
        split="validation",
    )

    print("train docs: ", len(train_docs))
    print("val docs: ", len(val_docs))

    task_module = SpanClassificationWithGazetteerEmbeddingTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        alias_to_wikidata_id_path=alias_to_wikidata_id_path,
        wikidata_id_to_index_path=wikidata_id_to_index_path,
        max_length=128,
    )

    task_module.prepare(train_docs)

    train_dataset = task_module.encode(train_docs, encode_target=True)
    val_dataset = task_module.encode(val_docs, encode_target=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_module.collate,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_module.collate,
    )

    model = SpanClassificationWithGazetteerEmbeddingModel(
        model_name_or_path=model_name_or_path,
        num_classes=len(task_module.label_to_id),
        t_total=len(train_dataloader) * num_epochs,
        additional_embeddings_path=additional_embeddings_path,
        tokenizer_vocab_size=len(task_module.tokenizer),
        learning_rate=1e-5,
        task_learning_rate=1e-5,
        freeze_model=False,
        layer_mean=False,
        use_mlp=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        dirpath=model_output_path,
        filename="multiconer-en-{epoch:02d}-val_f1-{val/f1:.2f}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
        save_weights_only=True,
    )

    task_module.save_pretrained(model_output_path)

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=num_epochs,
        gpus=1,
        # gpus=0,
        # checkpoint_callback=True,
        checkpoint_callback=False,
        # callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        # precision=32,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # trainer.save_checkpoint(model_output_path + "model.ckpt")
    # or
    # model.save_pretrained(model_output_path)


if __name__ == "__main__":
    main()
