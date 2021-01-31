from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from transformers import EvalPrediction, PreTrainedTokenizer
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


def build_calc_metrics_fn(tokenizer: PreTrainedTokenizer):
    rouge = Rouge(metrics=["rouge-2", "rouge-l"])

    def _compute_metrics(model_out: EvalPrediction):
        labels_ids = model_out.label_ids
        labels_ids[labels_ids == PAD_LABEL] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        predicted_str = tokenizer.batch_decode(model_out.predictions, skip_special_tokens=True)
        del model_out, labels_ids
        metrics_out = dict()

        # corpus_bleu requires references in the form of List[List[str]]
        references = list(map(lambda x: [x], label_str))
        metrics_out["bleu"] = corpus_bleu(references, predicted_str)
        del references

        rouge_scores = rouge.get_scores(predicted_str, label_str, avg=True, ignore_empty=True)
        # {'rouge-1': {'f': 0.383, 'p': 0.371, 'r': 0.403}, 'rouge-2': {...}, 'rouge-l': {...}}

        for rouge_type, rouge_metrics in rouge_scores.items():
            for name, value in rouge_metrics.items():
                complete_name = f"{rouge_type}_{name}"
                metrics_out[complete_name] = value

        return metrics_out

    return _compute_metrics
