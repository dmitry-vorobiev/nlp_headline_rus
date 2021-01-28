# nlp_headline_rus
News headline (title) generation using neural nets, trained on RIA dataset (russian lang)

## Installation

### Docker

```shell
docker build . --tag headlines:latest
```

## Usage

### Docker

```shell
docker run -it --name headlines_gpu --gpus all headlines:latest
```

inside docker:
```shell
sh /app/scripts/eval_docker.sh
```

## Pretrained weights

1. [URL](https://huggingface.co/dmitry-vorobiev/rubert_ria_headlines/tree/e0a2e3bf4a4c9069bb6cdf48ef7cc7f3301de4c6) | 
    bert2bert, initialized from `DeepPavlov/rubert-base-cased` and 
   fine-tuned on first 90% of [ria-new dataset](https://github.com/RossiyaSegodnya/ria_news_dataset) for 1.6 epochs.
