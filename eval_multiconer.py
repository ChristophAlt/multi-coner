import argparse
import bisect
import logging
import os
from itertools import accumulate
from typing import List, Tuple

from pytorch_ie import Pipeline
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.datamodules.datasets.multiconer import load_multiconer
from src.models.span_clf_with_gazetteer import SpanClassificationWithGazetteerModel
from src.taskmodules.span_clf_with_gazetteer import SpanClassificationWithGazetteerTaskModule


def seqeval_score(documents, predict_field: str = "entities", filter_entities: bool = False):
    true_tags = []
    pred_tags = []
    for doc in documents:
        _, doc_pred_tags = to_annotated_tokens(
            doc,
            predict_field=predict_field,
            filter_entities=filter_entities,
            use_annotations=False,
        )
        _, doc_true_tags = to_annotated_tokens(
            doc, predict_field=predict_field, filter_entities=filter_entities, use_annotations=True
        )

        true_tags.append(doc_true_tags)
        pred_tags.append(doc_pred_tags)

    results = classification_report(y_true=true_tags, y_pred=pred_tags, digits=4)
    return results


def filter_pred_entities(entities, is_flat_ner: bool = True):
    filtered_entities = []
    for ent in sorted(entities, key=lambda e: e.score, reverse=True):
        for f_ent in filtered_entities:
            if (ent.start < f_ent.start <= ent.end < f_ent.end) or (
                f_ent.start < ent.start <= f_ent.end < ent.end
            ):
                break

            if is_flat_ner and (
                ent.start <= f_ent.start <= f_ent.end <= ent.end
                or f_ent.start <= ent.start <= ent.end <= f_ent.end
            ):
                break

        else:
            filtered_entities.append(ent)

    return filtered_entities


def char_to_token_span(start_char: int, end_char: int, token_starts: List[int]) -> Tuple[int, int]:
    new_start = bisect.bisect_right(token_starts, start_char) - 1
    new_end = bisect.bisect_left(token_starts, end_char)

    return new_start, new_end


def to_annotated_tokens(
    document,
    predict_field: str = "entities",
    filter_entities: bool = True,
    use_annotations: bool = False,
):
    text = document.text

    if use_annotations:
        predictions = (
            filter_pred_entities(document.annotations("entities"), is_flat_ner=True)
            if filter_entities
            else document.annotations("entities")
        )
    else:
        predictions = (
            filter_pred_entities(document.predictions("entities"), is_flat_ner=True)
            if filter_entities
            else document.predictions("entities")
        )

    tokens = text.split(" ")

    token_starts = [0] + list(accumulate([len(token) + 1 for token in tokens]))

    tags = ["O"] * len(tokens)
    for prediction in predictions:
        start_idx, end_idx = char_to_token_span(prediction.start, prediction.end, token_starts)
        for i in range(start_idx, end_idx):
            prefix = "B-" if i == start_idx else "I-"
            tags[i] = prefix + prediction.label

    return tokens, tags


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    required=True,
    type=str,
    help="path containing the model to be evaluated",
)
parser.add_argument(
    "--dataset-dir",
    required=True,
    type=str,
    help="path containing the dataset to be evaluated",
)
parser.add_argument(
    "--dataset-split",
    required=True,
    type=str,
    choices=["train", "validation", "test"],
    help="dataset split to use for prediction",
)
parser.add_argument(
    "--pred-file",
    required=False,
    type=str,
    default=None,
    help="path to store the predictions",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="the batch size to use",
)
parser.add_argument(
    "--cuda-device",
    type=int,
    default=-1,
    help="the cuda device to use",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="enable debug logging",
)
parser.add_argument(
    "--filter-entities",
    action="store_true",
    default=False,
    help="filter overlapping entities",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=False,
    default=None,
    help="load model from the checkpoint file in dataset dir",
)


def main():
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=log_level
    )

    # gazetteer_path = "/home/christoph/Projects/research/multi_coner/data/gazetteers-2/gazetteer-all.txt"

    taskmodule = SpanClassificationWithGazetteerTaskModule.from_pretrained(
        args.model_path, use_efficient_gazetteer=False
    )

    if args.checkpoint is None:
        model = SpanClassificationWithGazetteerModel.from_pretrained(args.model_path)
    else:
        model = SpanClassificationWithGazetteerModel.load_from_checkpoint(
            os.path.join(args.model_path, args.checkpoint)
        )

    model = model.eval()

    ner_pipeline = Pipeline(model=model, taskmodule=taskmodule, device=args.cuda_device)

    eval_documents = load_multiconer(
        data_dir=args.dataset_dir,
        name="en",
        split=args.dataset_split,
    )

    predict_field = "entities"

    ner_pipeline(eval_documents, predict_field=predict_field, batch_size=args.batch_size)

    print(
        seqeval_score(
            documents=eval_documents,
            predict_field=predict_field,
            filter_entities=args.filter_entities,
        )
    )

    if args.pred_file is not None:
        out_lines = []
        for doc in eval_documents:
            tokens, tags = to_annotated_tokens(
                doc, predict_field=predict_field, filter_entities=args.filter_entities
            )
            for _, tag in zip(tokens, tags):
                out_lines.append(tag + "\n")

            out_lines.append("\n")

        with open(args.pred_file, "w") as f:
            f.writelines(out_lines[:-1])


if __name__ == "__main__":
    main()
