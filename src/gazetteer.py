import logging

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

        labels = set()
        for val in self.dictionary.values():
            labels.update(val)

        self.label_to_id = {label: i for i, label in enumerate(sorted(labels))}
        self.num_labels = len(self.label_to_id)

        logger.debug("label_to_id: %s", self.label_to_id)
        logger.debug("num_labels: %d", self.num_labels)

    def lookup(self, text: str):
        return self.dictionary.get(text, set())
