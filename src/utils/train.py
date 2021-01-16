from transformers import PreTrainedTokenizer
from typing import Any, Dict


def create_preprocess_fn(tokenizer: PreTrainedTokenizer,
                         encoder_max_length=512,
                         decoder_max_length=128):
    """
    Mostly copied from:
    https://github.com/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb
    """
    def _preprocess_fn(batch: Dict[str, Any]):
        inputs = tokenizer(
            text=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )

        outputs = tokenizer(
            text=batch["title"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                           for labels in batch["labels"]]

        return batch

    return _preprocess_fn
