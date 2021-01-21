# Using these scripts as a reference:
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py
# https://github.com/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb

import logging
import os
import sys
import re
import transformers

from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel, \
    EvalPrediction, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from typing import Dict, Optional, Union

from utils.filesys import check_output_dir, save_json, write_txt_file
from utils.model import assert_all_frozen, freeze_embeds, freeze_params
from utils.train import PAD_LABEL, create_preprocess_fn

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
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=32,
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
    # TODO: do we use it?
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
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


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def update_model_config(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    c = model.config
    c.decoder_start_token_id = tokenizer.bos_token_id
    c.eos_token_id = tokenizer.eos_token_id
    c.pad_token_id = tokenizer.pad_token_id

    # c.vocab_size = c.decoder.vocab_size
    c.num_beams = 4
    c.max_length = 128
    c.min_length = 32
    c.no_repeat_ngram_size = 3
    c.early_stopping = True
    c.length_penalty = 2.0


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

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
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    cache_dir = model_args.cache_dir
    model_name = model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name,
        cache_dir=cache_dir,
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    dataset = (load_dataset('json',
                            data_files=[data_args.data_json],
                            split='train',
                            cache_dir=cache_dir)
               .map(clean_html_tags, input_columns='text')
               .train_test_split(test_size=data_args.eval_split, shuffle=False))

    preprocess_train = create_preprocess_fn(
        tokenizer, data_args.max_source_length, data_args.max_target_length)
    preprocess_eval = create_preprocess_fn(
        tokenizer, data_args.max_source_length, data_args.val_max_target_length)

    train_data = dataset['train'].map(
        preprocess_train,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        remove_columns=["text", "title"]
    ).shuffle(seed=training_args.seed)

    eval_data = dataset['test'].map(
        preprocess_eval,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size,
        remove_columns=["text", "title"])

    for d in [train_data, eval_data]:
        d.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask",
                     "labels"])

    rouge = load_metric("rouge")

    def compute_metrics(output: EvalPrediction):
        labels_ids = output.label_ids
        prediction_ids = output.predictions

        predicted_str = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        labels_ids[labels_ids == PAD_LABEL] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=predicted_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    if os.path.isdir(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_name, model_name, cache_dir=cache_dir)
        update_model_config(model, tokenizer)

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    all_metrics = dict()

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(
            model_path=model_name if os.path.isdir(model_name) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams or model.config.num_beams
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        output = trainer.predict(
            test_dataset=eval_data,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams or model.config.num_beams
        )
        metrics = output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    output.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                predictions = list(map(str.strip, predictions))
                write_txt_file(predictions, os.path.join(training_args.output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics


if __name__ == "__main__":
    main()
