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
