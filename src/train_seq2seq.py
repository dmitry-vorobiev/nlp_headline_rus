# Using these scripts as a reference:
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py
# https://github.com/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb

import logging
import os
import sys
import transformers

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderConfig, \
    EncoderDecoderModel, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import is_main_process
from transformers.training_args import ParallelMode
from typing import Dict, Union

from args import ModelArguments, DataTrainingArguments
from dataset_utils import build_datasets
from utils.filesys import check_output_dir, save_json, write_txt_file
from utils.model import assert_all_frozen, freeze_embeds, freeze_params
from train_utils import build_calc_metrics_fn

logger = logging.getLogger(__name__)


def handle_metrics(split: str, metrics: Dict[str, Union[int, float]], output_dir: str):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            value = round(value, 4)
        logger.info(f"  {key} = {value}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def update_model_config(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    c = model.config
    c.decoder_start_token_id = tokenizer.bos_token_id
    c.eos_token_id = tokenizer.eos_token_id
    c.pad_token_id = tokenizer.pad_token_id

    # c.vocab_size = c.decoder.vocab_size
    c.num_beams = 4
    c.max_length = 48
    c.min_length = 4
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
    model_name_or_path = model_args.model_name_or_path
    output_dir = training_args.output_dir
    do_train = training_args.do_train
    do_eval = training_args.do_eval
    do_predict = training_args.do_predict

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name_or_path,
        cache_dir=cache_dir,
    )
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    train_dataset, eval_dataset = build_datasets(data_args, tokenizer,
                                                 cache_dir=cache_dir,
                                                 skip_train=not do_train,
                                                 skip_eval=not (do_eval or do_predict))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_name_or_path,
        cache_dir=cache_dir,
    )

    if isinstance(config, EncoderDecoderConfig):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_name_or_path,
            decoder_pretrained_model_name_or_path=model_name_or_path,
            tie_encoder_decoder=model_args.tie_encoder_decoder,
            cache_dir=cache_dir)
        update_model_config(model, tokenizer)

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_calc_metrics_fn(tokenizer) if do_eval or do_predict else None,
        tokenizer=tokenizer,
    )
    all_metrics = dict()

    # Training
    if do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(
            model_path=model_name_or_path if os.path.isdir(model_name_or_path) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(output_dir)

    # Evaluation
    if do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams or model.config.num_beams
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, output_dir)
            all_metrics.update(metrics)

    if do_predict:
        logger.info("*** Predict ***")

        output = trainer.predict(
            test_dataset=eval_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams or model.config.num_beams
        )
        metrics = output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    output.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                predictions = list(map(str.strip, predictions))
                write_txt_file(predictions, os.path.join(output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(output_dir, "all_results.json"))

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
