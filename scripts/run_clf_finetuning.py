from dataclasses import dataclass, field
from filelock import FileLock
from typing import Optional, List

import logging
import os
import sys
import time

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)
from transformers import (
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    BartTokenizer,
    BartTokenizerFast
)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.processors.glue import (
    PreTrainedTokenizer,
    InputFeatures
)
import torch
from torch.utils.data.dataset import Dataset

from utils import (
    DataProcessor,
    convert_examples_to_features,
    Split,
    mode_compute_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeneralArguments:
    """Arguments for setting up general configuarion"""
    output_mode: str = field(
        metadata={"help": "indicating the output mode. Either ``regression`` or ``classification``"}
    )
    convert_to_tf: bool = field(
        default=True, metadata={"help": "Convert the final PyTorch model to TensorFlow!"}
    )
    labels: Optional[str] = field(
        default='', metadata={"help": "Enter the label name of your labels separate by comma: apple,orange,banana"}
    )

    def _labels(self):
        if self.output_mode == 'classification':
            return [str(l) for l in list(set(self.labels.split(',')))]

        return []


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class YourDataset(Dataset):
    """This will be superseded by a framework-agnostic approach soon."""

    args: DataTrainingArguments
    op_args: GeneralArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: DataTrainingArguments,
            op_args: GeneralArguments,
            tokenizer: PreTrainedTokenizer,
            limit_length: Optional[int] = None,
            mode: Split = Split.train,
            cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = DataProcessor()
        self.output_mode = op_args.output_mode

        self.processor.set_labels(op_args._labels())

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            )
        )

        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                RobertaTokenizer,
                RobertaTokenizerFast,
                XLMRobertaTokenizer,
                BartTokenizer,
                BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = self.processor.get_labels()

                if mode.value == 'train':
                    examples = self.processor.get_train_examples(args.data_dir)
                elif mode.value == 'dev':
                    examples = self.processor.get_test_examples(args.data_dir)
                elif mode.value == 'test':
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = None

                if limit_length is not None:
                    examples = examples[:limit_length]

                self.features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    task=args.task_name,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


def main():
    parser = HfArgumentParser((GeneralArguments, ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        op_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        op_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Global setup
    labels = op_args._labels()
    output_mode = op_args.output_mode

    if output_mode == 'classification':
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {v: k for k, v in label2id.items()}
    else:
        num_labels = 1

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if output_mode == 'classification':
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        YourDataset(data_args, op_args=op_args, tokenizer=tokenizer, mode=Split.train)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        YourDataset(data_args, op_args=op_args, tokenizer=tokenizer, mode=Split.dev)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        YourDataset(data_args, op_args=op_args, tokenizer=tokenizer, mode=Split.test)
        if training_args.do_predict
        else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=mode_compute_metrics(op_args.output_mode),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Convert PT to TF
    if op_args.convert_to_tf:
        logger.info("***** Convert PT to TF {} *****".format(training_args.output_dir + 'pytorch_model.bin'))
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(training_args.output_dir, from_pt=True)
        tf_model.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")

        test_datasets = [test_dataset]
        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions

            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
