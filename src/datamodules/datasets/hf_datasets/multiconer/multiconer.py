import os
import re
import typing as T
from posixpath import split

import datasets
from conllu.models import TokenList
from conllu.parser import (
    _FieldParserType,
    _MetadataParserType,
    parse_conllu_plus_fields,
    parse_sentences,
    parse_token_and_metadata,
)

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
"""

_DESCRIPTION = """\
"""


_DATA_DIRS = {
    "en": "EN-English",
    "de": "DE-German",
}


def _parse_incr(
    in_file: T.TextIO,
    fields: T.Optional[T.Sequence[str]] = None,
    field_parsers: T.Dict[str, _FieldParserType] = None,
    metadata_parsers: T.Optional[T.Dict[str, _MetadataParserType]] = None,
) -> T.Iterator[TokenList]:
    if not hasattr(in_file, "read"):
        raise FileNotFoundError("Invalid file, 'parse_incr' needs an opened file as input")

    if not fields:
        fields = parse_conllu_plus_fields(in_file, metadata_parsers=metadata_parsers)

    for sentence in parse_sentences(in_file):
        sentence = re.sub(" ", "  ", sentence)
        yield parse_token_and_metadata(
            sentence, fields=fields, field_parsers=field_parsers, metadata_parsers=metadata_parsers
        )


class MultiCoNERConfig(datasets.BuilderConfig):
    """BuilderConfig for MultiCoNER"""

    def __init__(self, **kwargs):
        """BuilderConfig for MultiCoNER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiCoNERConfig, self).__init__(**kwargs)


class MultiCoNER(datasets.GeneratorBasedBuilder):
    """MultiCoNER dataset."""

    BUILDER_CONFIGS = [
        MultiCoNERConfig(
            name="en", version=datasets.Version("1.0.0"), description="English MultiCoNER dataset"
        ),
        MultiCoNERConfig(
            name="de", version=datasets.Version("1.0.0"), description="German MultiCoNER dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-LOC",
                                "I-LOC",
                                "B-GRP",
                                "I-GRP",
                                "B-CORP",
                                "I-CORP",
                                "B-PROD",
                                "I-PROD",
                                "B-CW",
                                "I-CW",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    @property
    def manual_download_instructions(self):
        return ""

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('multiconer', data_dir=...)` that includes the unzipped files. Manual download instructions: {}".format(
                    data_dir, self.manual_download_instructions
                )
            )

        split_generators = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, _DATA_DIRS[self.config.name], self.config.name + "_train.conll"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, _DATA_DIRS[self.config.name], self.config.name + "_dev.conll"
                    ),
                },
            ),
        ]

        test_filepath = os.path.join(
            data_dir, _DATA_DIRS[self.config.name], self.config.name + "_test.conll"
        )

        if os.path.isfile(test_filepath):
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": test_filepath,
                    },
                )
            )

        return split_generators

    def _generate_examples(self, filepath):
        def split_comment(key, value):
            if "domain" not in key:
                return []

            id_string = re.split("\t", key)[0]
            id_ = re.split(" +", id_string)[1]

            return [("id", id_), ("domain", value)]

        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for sentence in parse_sentences(f):
                guid = None
                tokens = []
                ner_tags = []
                for i, line in enumerate(sentence.split("\n")):
                    line = line.strip()
                    if i == 0:
                        assert line.startswith("#")
                        line = line.strip("#")
                        id_string = re.split("\t", line)[0]
                        guid = re.split(" ", id_string.strip())[1]
                        continue

                    parts = line.split(" ")
                    tokens.append(parts[0])

                    # labels are missing
                    if len(parts) == 3:
                        ner_tags.append("O")
                    else:
                        ner_tags.append(parts[3])

                    # guid = tokenlist.metadata["id"]
                    # tokens = [token["form"] for token in tokenlist]
                    # ner_tags = [token["tag"] for token in tokenlist]

                assert guid is not None

                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }

            # for tokenlist in _parse_incr(
            #     f,
            #     fields=["form", "pos", "chunk", "tag"],
            #     metadata_parsers={"__fallback__": split_comment},
            # ):
            #     guid = tokenlist.metadata["id"]
            #     tokens = [token["form"] for token in tokenlist]
            #     ner_tags = [token["tag"] for token in tokenlist]

            #     yield guid, {
            #         "id": str(guid),
            #         "tokens": tokens,
            #         "ner_tags": ner_tags,
            #     }
