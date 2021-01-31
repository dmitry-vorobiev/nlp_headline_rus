import logging
import os
import re
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer
from typing import Dict, Union

from args import DataTrainingArguments
from train_utils import create_preprocess_fn

logger = logging.getLogger(__name__)


def build_datasets(data_args: DataTrainingArguments,
                   tokenizer: PreTrainedTokenizer,
                   cache_dir=None) -> Union[Dataset, DatasetDict]:
    json_path = data_args.data_json
    data_dir = data_args.load_data_from

    if json_path is not None:
        logger.info("Preprocessing new dataset from {}".format(json_path))

        dataset = (load_dataset('json', data_files=[json_path], split='train', cache_dir=cache_dir)
                   .map(clean_html_tags, input_columns='text')
                   .train_test_split(test_size=data_args.eval_split, shuffle=False))

        proc_train = create_preprocess_fn(
            tokenizer, data_args.max_source_length, data_args.max_target_length)
        proc_eval = create_preprocess_fn(
            tokenizer, data_args.max_source_length, data_args.val_max_target_length)
        prep_bs = data_args.tokenizer_batch_size

        map_kwargs = dict(batched=True, batch_size=prep_bs, remove_columns=["text", "title"])
        dataset["train"] = dataset["train"].map(proc_train, **map_kwargs)
        dataset["test"] = dataset["test"].map(proc_eval, **map_kwargs)
        dataset.set_format(type="torch",
                           columns=["input_ids", "attention_mask", "decoder_input_ids",
                                    "decoder_attention_mask", "labels"])

        if data_args.save_data_to is not None:
            save_dir = data_args.save_data_to
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            logger.info("Saving preprocessed dataset to {}".format(save_dir))
            dataset.save_to_disk(save_dir)

    elif data_dir is not None:
        logger.info("Loading preprocessed dataset from {}".format(data_dir))
        dataset = load_from_disk(data_dir)
        assert "train" in dataset.keys()
        assert "test" in dataset.keys()
    else:
        raise AttributeError("You must provide either `--data_json` or `--load_data_from` argument.")

    return dataset


def clean_html_tags(s: str) -> Dict[str, str]:
    s = re.sub('</?[\w\W]+?>', ' ', s)
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
