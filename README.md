# nlp_headline_rus
News headline generation using neural nets, trained on ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset).

## Task evaluation
Using last 10% of the ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) for now. May switch to another dataset later.

## Installation

### PyPI

```shell
pip install -r requirements.txt
```

### Docker

```shell
docker build . --tag headlines:latest
```

## Usage

### Docker

Build docker image:

```shell
docker run -it --name headlines_gpu --gpus all headlines:latest
```

Run evaluation script inside docker container:
```shell
sh /app/scripts/eval_docker.sh
```

By default, results can be found at `/io/output`.

## Pretrained weights

1. [URL](https://huggingface.co/dmitry-vorobiev/rubert_ria_headlines/tree/e0a2e3bf4a4c9069bb6cdf48ef7cc7f3301de4c6) | 
   2021-01-28 | bert2bert, initialized with the `DeepPavlov/rubert-base-cased` pretrained weights and 
   fine-tuned on the first 90% of ["Rossiya Segodnya" news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) for 1.6 epochs.
