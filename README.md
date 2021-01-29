# nlp_headline_rus
News headline generation using neural nets, trained on ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset).

## Task evaluation
Using last 10% of the ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) for now. May switch to another dataset later.

## Installation

### PyPI

You may want to create a virtual environment first

```shell
pip install -r requirements.txt
```

### Docker

Build docker image

```shell
docker build . --tag headlines:latest
```

## Usage

### In code

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "dmitry-vorobiev/rubert_ria_headlines"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

text = "Скопируйте текст статьи / новости"

encoded_batch = tokenizer.prepare_seq2seq_batch(
    [text],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512)

output_ids = model.generate(
    input_ids=encoded_batch["input_ids"],
    max_length=32,
    no_repeat_ngram_size=3,
    num_beams=5,
    top_k=0
)

headline = tokenizer.decode(output_ids[0], 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False)
print(headline)
```

### Docker

Run docker container interactive session:

```shell
docker run -it --name headlines_gpu --gpus all headlines:latest
```

Execute evaluation script:
```shell
sh /app/scripts/eval_docker.sh
```

By default, results can be found at `/io/output`.

## Pretrained weights

1. [URL](https://huggingface.co/dmitry-vorobiev/rubert_ria_headlines/tree/e0a2e3bf4a4c9069bb6cdf48ef7cc7f3301de4c6) | 
   2021-01-28 | bert2bert, initialized with the `DeepPavlov/rubert-base-cased` pretrained weights and 
   fine-tuned on the first 90% of ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) for 1.6 epochs.
