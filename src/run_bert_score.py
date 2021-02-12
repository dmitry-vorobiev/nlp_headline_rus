import logging
import csv
import os
import sys
import transformers
from bert_score import score
from datasets import load_dataset
from transformers import HfArgumentParser

from args import EvaluationArguments
from dataset_utils import deduce_dataset_args
from utils.filesys import save_json

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(EvaluationArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args: EvaluationArguments = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args: EvaluationArguments = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    tr_log = transformers.utils.logging
    tr_log.set_verbosity_info()
    tr_log.enable_default_handler()
    tr_log.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", args)

    output_dir = args.output_dir

    data_kwargs = dict(cache_dir=args.cache_dir, split="train")
    test_data_kwargs = deduce_dataset_args(args.test_data)
    test_data = load_dataset(**test_data_kwargs, **data_kwargs)

    if args.test_size < 1:
        test_data = test_data.train_test_split(test_size=args.test_size, shuffle=False)["test"]

    ref_col = args.ref_col
    references = test_data[ref_col]

    predictions_kwargs = deduce_dataset_args(args.predictions)
    predictions = load_dataset(**predictions_kwargs, **data_kwargs)["text"]

    if len(test_data) != len(predictions):
        raise AttributeError(
            "Number of predictions doesn't match length of test dataset: "
            "{} vs {}".format(len(test_data), len(predictions)))

    precision, recall, f1_measure = score(
        cands=predictions,
        refs=references,
        lang="ru",
        verbose=True,
        batch_size=args.batch_size,
        nthreads=args.num_workers,
    )

    metrics = dict(
        min_precision=precision.min().item(),
        avg_precision=precision.mean().item(),
        min_recall=recall.min().item(),
        avg_recall=recall.mean().item(),
        min_f1_score=f1_measure.min().item(),
        avg_f1_score=f1_measure.mean().item()
    )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info(f"***** BERT score *****")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            value = round(value, 5)
        logger.info(f"  {key} = {value}")
    save_json(metrics, os.path.join(output_dir, f"bert_score.json"))

    path = os.path.join(output_dir, f"bert_score_all.csv")
    logger.info("Saving individual results to {}".format(path))

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["prediction", "reference", "precision", "recall", "f1-score"])

        for i, (predicted_str, reference_str) in enumerate(zip(predictions, references)):
            writer.writerow([
                predicted_str,
                reference_str,
                precision[i].item(),
                recall[i].item(),
                f1_measure[i].item()
            ])

    print("DONE")


if __name__ == "__main__":
    main()
