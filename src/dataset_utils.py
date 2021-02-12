import logging
import os
import re
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Tuple

from args import DataTrainingArguments

PAD_LABEL = -100
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
    break_token = data_args.line_break_token
    train_data, eval_data = None, None
    dataset = DatasetDict()

    if add_line_breaks:
        tokenizer.add_special_tokens(dict(additional_special_tokens=[break_token]))

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

        if not data_args.skip_text_clean:
            normalize = partial(normalize_text, add_line_breaks=add_line_breaks, brk=break_token)
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

    if "train" in dataset:
        train_data = dataset["train"]
    if "test" in dataset:
        eval_data = dataset["test"]
    return train_data, eval_data


def normalize_text(s: str, add_line_breaks=False, brk="[BRK]") -> Dict[str, str]:
    if add_line_breaks:
        # </p> <p> | <br> | \n
        s = re.sub('</p>(\W)?(<p>)?|<br\s*?/?>|\n', brk, s)
    s = re.sub('</?[\w\W]+?>', ' ', s)
    s = re.sub('\n|&nbsp;', ' ', s)
    s = re.sub('&[mn]?dash;', ' – ', s)
    s = re.sub('&lt;', '<', s)
    s = re.sub('&gt;', '>', s)
    s = re.sub('&rsquo;', '’', s)
    s = re.sub('&hellip;', '…', s)
    s = re.sub('&amp;', '&', s)
    if add_line_breaks:
        # remove repeating tokens
        s = re.sub('(\[' + brk[1:-1] + ']\s*){2,}', brk, s)
        s = s.strip().strip(brk)
    s = re.sub('\s{2,}', ' ', s)
    return dict(text=s.strip())


def deduce_dataset_args(data_path: str) -> Dict[str, Any]:
    out = dict(path=data_path)

    if os.path.exists(data_path) and os.path.isfile(data_path):
        _, ext = os.path.splitext(data_path)
        out = dict(data_files=[data_path])

        if ext.lower() in [".json", ".jsonl"]:
            out["path"] = "json"
        elif ext.endswith("txt"):
            out["path"] = "text"
        else:
            raise ValueError("Unregistered file extension: {}".format(ext))

    return out


def create_preprocess_fn(tokenizer: PreTrainedTokenizer,
                         max_input_len=512,
                         max_output_len=128):
    """
    Mostly copied from:
    https://github.com/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb
    """
    def _preprocess_fn(batch: Dict[str, Any]):
        inputs = tokenizer(
            text=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_input_len,
        )

        outputs = tokenizer(
            text=batch["title"],
            padding="max_length",
            truncation=True,
            max_length=max_output_len,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        pad_token = tokenizer.pad_token_id
        batch["labels"] = [[PAD_LABEL if token == pad_token else token for token in labels]
                           for labels in batch["labels"]]

        return batch

    return _preprocess_fn
