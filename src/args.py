import os
from dataclasses import dataclass, field
from typing import Dict, Optional

cur_dir = os.path.dirname(__file__)
default_cache_dir = os.path.join(cur_dir, ".cache")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="DeepPavlov/rubert-base-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=default_cache_dir,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether to freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    tie_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to share weights between encoder/decoder parts of the model."}
    )


@dataclass
class DataTrainingArguments:
    data_json: Optional[str] = field(
        default=None,
        metadata={"help": "The input .json file"}
    )
    save_data_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "Save preprocessed dataset to this directory"
        }
    )
    load_data_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load preprocessed dataset from this directory"
        }
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=48,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=48,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. "
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    eval_split: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Fraction of dataset or # of samples to use for evaluation."
        }
    )
    tokenizer_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for text tokenization"
        }
    )
    n_train: Optional[int] = field(
        default=-1,
        metadata={"help": "# training examples. -1 means use all."}
    )
    n_val: Optional[int] = field(
        default=-1,
        metadata={"help": "# validation examples. -1 means use all."}
    )
    n_test: Optional[int] = field(
        default=-1,
        metadata={"help": "# test examples. -1 means use all."}
    )
    eval_beams: Optional[int] = field(
        default=3,
        metadata={"help": "# num_beams to use for evaluation."}
    )
    add_line_breaks: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use special token to identify line breaks in news text"},
    )
    line_break_token: Optional[str] = field(
        default="[unused99]",
        metadata={
            "help": "String representation of line break token"
        }
    )
    # TODO: do we use it?
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
