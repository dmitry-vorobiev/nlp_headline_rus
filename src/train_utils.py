from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from transformers import EvalPrediction, PreTrainedTokenizer

from dataset_utils import PAD_LABEL


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
