import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from src.datamodules.datasets.multiconer import load_multiconer
from src.models.span_clf_with_gazetteer import SpanClassificationWithGazetteerModel
from src.taskmodules.span_clf_with_gazetteer import SpanClassificationWithGazetteerTaskModule


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
                lr=1e-4,
                train_bn=False,
            )

            # for group in optimizer.param_groups:
            #     group["lr"] = 1e-5


def main():
    log_level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=log_level
    )

    pl.seed_everything(42)

    model_output_path = "/media/christoph/HDD/models/multiconer-en-spanclf-wiki-gazetteer-aug-4/"
    wiki_to_vec_file = "/home/christoph/Downloads/enwiki_20180420_100d.kv"
    # gazetteer_path = "/home/christoph/Downloads/gazetteers"

    gazetteer_path = "/home/christoph/Projects/research/notebooks/gazetteers/gaz_combined_3.txt"

    # tokenizer_name_or_path = "bert-base-uncased"
    tokenizer_name_or_path = "google/electra-large-discriminator"
    model_name_or_path = tokenizer_name_or_path
    num_epochs = 100
    batch_size = 32

    data_dir = "/home/christoph/Downloads/training_data/"
    # data_dir = "/home/christoph/Projects/research/multi_coner/notebooks/augmented_multiconer/improved_validation/"

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

    task_module = SpanClassificationWithGazetteerTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        wiki_to_vec_file=wiki_to_vec_file,
        gazetteer_path=gazetteer_path,
        use_efficient_gazetteer=False,
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

    model = SpanClassificationWithGazetteerModel(
        model_name_or_path=model_name_or_path,
        num_classes=len(task_module.label_to_id),
        t_total=len(train_dataloader) * num_epochs,
        wiki_to_vec_file=wiki_to_vec_file,
        num_gazetteer_labels=task_module.gazetteer.num_labels,
        tokenizer_vocab_size=len(task_module.tokenizer),
        tokenizer_unk_token_id=task_module.tokenizer.mask_token_id,
        augment_input=True,
        augment_mask_probability=0.2,
        learning_rate=1e-5,
        task_learning_rate=1e-4,
        freeze_model=False,
        layer_mean=False,
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
        checkpoint_callback=True,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # trainer.save_checkpoint(model_output_path + "model.ckpt")
    # or
    # model.save_pretrained(model_output_path)


if __name__ == "__main__":
    main()
