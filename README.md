# talm 

Welcome to talm, short for train-a-language-model. 

This is a project I created for training your own language model from zero. 

By "from zero" I mean training your own tokenizer, downloading a dataset, and 
pre-training a transformer model.

If you'd like to go through the process of training a transformer-based language model, 
you can follow the instructions below!

Contents:

- [Setup](#1-setup-the-environment) 

## 1. Setup The Environment

First, you'll need to setup your local environment. This project manages dependencies 
with [rye](https://rye.astral.sh/), therefore you will need to install it to proceed. 

Check out [the instructions](https://rye.astral.sh/guide/installation/) for more.

Once you have installed it, in the project repo run:

```rye sync```

This will create a virtual environment in `./venv`. Now, you're all set!

## 2. Train a tokenizer
**Note**: This is not mandatory, a very minimal tokenizer I have trained is located in `tokenizers/base.json`, 
and is sufficient to continue with training your own model. You can [skip](#3-download-a-dataset) this step if you would like.
To start, you will train your own tokenizer using the [`tokencoder`](https://github.com/lucspt/tokencoder) package.

To do so, you just have to run a python script:

```text
rye run train_tokenizer [OPTIONS]
```



## 3. Download A Dataset

Once you have setup your environment, you'll need to download a dataset 
to train your model on. 

You can use rye to run a python script that will download and shard a
Huggingface dataset like so:

```text
rye run download_data <ds-name> [OPTIONS]
```

Currently the supported datasets are [`fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 
and [`smol`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). 

