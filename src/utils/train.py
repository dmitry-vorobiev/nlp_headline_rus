from transformers import PreTrainedTokenizer
from typing import Any, Dict

PAD_LABEL = -100


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
