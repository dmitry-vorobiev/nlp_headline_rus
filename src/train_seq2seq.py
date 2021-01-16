# Using these scripts as a reference:
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py
# https://github.com/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb

import logging
import os
import sys
import re
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel, HfArgumentParser, set_seed
from typing import Dict, Optional

from utils.train import create_preprocess_fn

cur_dir = os.path.dirname(__file__)
default_cache_dir = os.path.join(cur_dir, ".cache")

logger = logging.getLogger(__name__)


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
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    data_json: str = field(
        metadata={"help": "The input .json file"}
    )


def clean_html_tags(s: str) -> Dict[str, str]:
    s = re.sub('</?[\w\W]+?>', ' TAG ', s)
    s = re.sub('\n|&nbsp;', ' ', s)
    s = re.sub('&[mn]?dash;', ' – ', s)
    s = re.sub('&lt;', '<', s)
    s = re.sub('&gt;', '>', s)
    # s = re.sub(r"\'", "'", s)
    s = re.sub('&rsquo;', '’', s)
    s = re.sub('&hellip;', '…', s)
    s = re.sub('&amp;', '&', s)
    s = re.sub('\s\s+', ' ', s)
    return dict(text=s)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    cache_dir = model_args.cache_dir
    model_name = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name,
        cache_dir=cache_dir,
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    preprocess_inputs = create_preprocess_fn(tokenizer)

    dataset = load_dataset(
        'json',
        data_files=[data_args.data_json],
        split='train',
        cache_dir=cache_dir
    ).map(clean_html_tags, input_columns='text')

    train_data = dataset.map(
        preprocess_inputs,
        batched=True,
        batch_size=4,
        remove_columns=["text", "title"]
    )

    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask",
                 "labels"],
    )

    seq2seq = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name, model_name, cache_dir=cache_dir)

    inputs = train_data[:3]
    outputs = seq2seq(**inputs)

    print(outputs['logits'].shape)


if __name__ == "__main__":
    main()
