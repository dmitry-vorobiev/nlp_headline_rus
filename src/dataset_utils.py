import logging
import os
import re
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from functools import partial
from transformers import PreTrainedTokenizer, AddedToken
from typing import Dict, Tuple

from args import DataTrainingArguments
from train_utils import create_preprocess_fn

BREAK_TOKEN_STRIPPED = 'BRK'
BREAK_TOKEN = f'[{BREAK_TOKEN_STRIPPED}]'

logger = logging.getLogger(__name__)


def build_datasets(
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        cache_dir=None,
        skip_train=False,
        skip_eval=False) -> Tuple[Dataset, Dataset]:
    if skip_eval and skip_train:
        logger.warning("Both `skip_train` and `skip_eval` are set to True")

    json_path = data_args.data_json
    data_dir = data_args.load_data_from
    add_line_breaks = data_args.add_line_breaks
    train_data, eval_data = None, None
    dataset = DatasetDict()

    if json_path is not None:
        logger.info("Preprocessing new dataset from {}".format(json_path))
        eval_split = data_args.eval_split
        save_dir = data_args.save_data_to

        dataset = load_dataset('json', data_files=[json_path], cache_dir=cache_dir)
        if eval_split < 1:
            dataset = dataset["train"].train_test_split(test_size=eval_split, shuffle=False)

        if save_dir is None:
            # Spend less time on preprocessing
            if skip_train:
                del dataset["train"]
            if skip_eval and "test" in dataset:
                del dataset["test"]

        normalize = partial(normalize_text, add_line_breaks=add_line_breaks)
        dataset = dataset.map(normalize, input_columns='text')

        proc_kwargs = dict(
            batched=True,
            batch_size=data_args.tokenizer_batch_size,
            remove_columns=["text", "title"])

        if "train" in dataset:
            proc_train = create_preprocess_fn(
                tokenizer, data_args.max_source_length, data_args.max_target_length)
            dataset["train"] = dataset["train"].map(proc_train, **proc_kwargs)

        if "test" in dataset:
            proc_eval = create_preprocess_fn(
                tokenizer, data_args.max_source_length, data_args.val_max_target_length)
            dataset["test"] = dataset["test"].map(proc_eval, **proc_kwargs)

        dataset.set_format(type="torch",
                           columns=["input_ids", "attention_mask", "decoder_input_ids",
                                    "decoder_attention_mask", "labels"])

        save_dir = data_args.save_data_to
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            logger.info("Saving preprocessed dataset to {}".format(save_dir))
            dataset.save_to_disk(save_dir)

    elif data_dir is not None:
        logger.info("Loading preprocessed dataset from {}".format(data_dir))
        if skip_train:
            eval_data = load_from_disk(os.path.join(data_dir, "test"))
        elif skip_eval:
            train_data = load_from_disk(os.path.join(data_dir, "train"))
        else:
            dataset = load_from_disk(data_dir)
    else:
        raise AttributeError("You must provide either `--data_json` or `--load_data_from` argument.")

    if add_line_breaks:
        tokenizer.add_special_tokens(dict(additional_special_tokens=[BREAK_TOKEN]))

    if "train" in dataset:
        train_data = dataset["train"]
    if "test" in dataset:
        eval_data = dataset["test"]
    return train_data, eval_data


def normalize_text(s: str, add_line_breaks=False) -> Dict[str, str]:
    if add_line_breaks:
        # </p> <p> | <br> | \n
        s = re.sub('</p>(\W)?(<p>)?|<br\s*?/?>|\n', BREAK_TOKEN, s)
    s = re.sub('</?[\w\W]+?>', ' ', s)
    s = re.sub('\n|&nbsp;', ' ', s)
    s = re.sub('&(m|n)?dash;', ' – ', s)
    s = re.sub('&lt;', '<', s)
    s = re.sub('&gt;', '>', s)
    s = re.sub('&rsquo;', '’', s)
    s = re.sub('&hellip;', '…', s)
    s = re.sub('&amp;', '&', s)
    if add_line_breaks:
        # remove repeating tokens
        s = re.sub('(\[' + BREAK_TOKEN_STRIPPED + ']\s*){2,}', BREAK_TOKEN, s)
        s = s.strip().strip(BREAK_TOKEN)
    s = re.sub('\s{2,}', ' ', s)
    return dict(text=s.strip())
