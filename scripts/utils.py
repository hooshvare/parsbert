from enum import Enum
from typing import Dict, Optional, Union, List

import logging
import os

import numpy as np

from sklearn.metrics import f1_score, mean_squared_error, r2_score, mean_absolute_error

from transformers import (
    EvalPrediction
)
from transformers.data.processors.glue import (
    PreTrainedTokenizer,
    DataProcessor as MainDataProcessor,
    InputExample,
    InputFeatures
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(MainDataProcessor):
    """Processor for converting your dataset into sequence classification dataset."""
    _labels = []

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensors."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def set_labels(self, labels):
        self._labels = labels

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return list(self._labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                idx_text = line.index('text')
                idx_label = line.index('label')
            else:
                guid = "%s-%s" % (set_type, i)
                text_a = line[idx_text]
                label = line[idx_label]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def _convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None):
    if max_length is None:
        max_length = tokenizer.max_len

    logger.info("Using label list %s for task %s" % (label_list, task))
    logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None):
    return _convert_examples_to_features(
        examples,
        tokenizer,
        max_length=max_length,
        task=task,
        label_list=label_list,
        output_mode=output_mode
    )


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def simple_accuracy(predictions, labels):
    return (predictions == labels).mean()


def acc_and_f1(predictions, labels):
    acc = simple_accuracy(predictions, labels)
    f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def mse_and_mae_and_r2(predictions, labels):
    return {
        "mse": mean_squared_error(labels, predictions),
        "mae": mean_absolute_error(labels, predictions),
        "r2": r2_score(labels, predictions),
    }


def _cls_compute_metrics(predictions, labels):
    return acc_and_f1(predictions, labels)


def _rgr_compute_metrics(predictions, labels):
    return acc_and_f1(predictions, labels)


def cls_compute_metrics(p: EvalPrediction) -> Dict:
    predictions = np.argmax(p.predictions, axis=1)
    return _cls_compute_metrics(predictions, p.label_ids)


def rgr_compute_metrics(p: EvalPrediction) -> Dict:
    return _rgr_compute_metrics(p.predictions, p.label_ids)


def mode_compute_metrics(mode):
    if mode == 'classification':
        return cls_compute_metrics
    elif mode == 'regression':
        return rgr_compute_metrics
    else:
        raise ValueError("the mode does not exist in ['classification', 'regression']")
