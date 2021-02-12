import logging
import csv
import os
import sys
import transformers
from bert_score import score
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import HfArgumentParser
from typing import Optional

from args import DataTrainingArguments, default_cache_dir
from utils.filesys import save_json, write_txt_file

logger = logging.getLogger(__name__)


@dataclass
class DataEvaluationArguments(DataTrainingArguments):
    predictions_txt: str = field(
        default=None,
        metadata={"help": "Path to generated predictions .txt file"}
    )
    cache_dir: Optional[str] = field(
        default=default_cache_dir,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    output_dir: str = field(
        default="./bert_score",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    eval_split: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Fraction of dataset or # of samples to use for evaluation."
        }
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU"}
    )
    num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of threads to use in bert_score.score()"
        },
    )


def main():
    parser = HfArgumentParser(DataEvaluationArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args: DataEvaluationArguments = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args: DataEvaluationArguments = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", args)

    cache_dir = args.cache_dir
    output_dir = args.output_dir

    data_kwargs = dict(cache_dir=cache_dir, split="train")
    dataset = load_dataset('json', data_files=[args.data_json], **data_kwargs)
    dataset = dataset.train_test_split(test_size=args.eval_split, shuffle=False)
    test_data = dataset["test"]

    predictions = load_dataset("text", data_files=[args.predictions_txt], **data_kwargs)

    assert len(test_data) == len(predictions)

    precision, recall, f1_measure = score(
        cands=predictions["text"],
        refs=test_data["title"],
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

        for i, (prediction, reference) in enumerate(zip(predictions["text"], test_data["title"])):
            writer.writerow([
                prediction,
                reference,
                precision[i].item(),
                recall[i].item(),
                f1_measure[i].item()
            ])

    print("DONE")


if __name__ == "__main__":
    main()
