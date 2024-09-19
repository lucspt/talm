# talm 

Welcome to `talm`, short for train a language model. 

This is a project I created with scripts for training your own language model "from scratch". 

By "from scratch" I mean training your own tokenizer, downloading a dataset, and 
pre-training a transformer model.

If you'd like to go through the process of training a transformer-based language model, 
you can follow the instructions below!

Steps:

1. [Setup](#1-setup-the-environment) 
2. [Train a tokenizer](#2-train-a-tokenizer)
3. [Download dataset](#3-download-a-dataset)
4. [Train a model](#4-train-a-model)

> [!NOTE]
There exists a c++ implementation of the language model in the [cpplm repo](https://github.com/lucspt/cpplm).

## 1. Setup The Environment

First, you'll need to setup your local environment. This project manages dependencies 
with [rye](https://rye.astral.sh/), therefore you will need to install it to proceed. 

Check out [the instructions](https://rye.astral.sh/guide/installation/) for more.

Once you have installed it, in the project repo run:

`rye sync`

This will create a virtual environment in the repo root. Now, you're all set!

## 2. Train a tokenizer
> [!NOTE]
This is not mandatory. If you would like to use a very minimal tokenizer I have already trained, [skip](#3-download-a-dataset) to the next step.

To start, you will train your own tokenizer using the [`tokencoder`](https://github.com/lucspt/tokencoder) package.

To do so, you just have to run a python script:

```text
rye run train_tokenizer [OPTIONS]
```

Here are the options:
```text
options:
  -vs VOCAB_SIZE, --vocab-size VOCAB_SIZE
                        The desired vocab size
  -n TOKENIZER_NAME, --name TOKENIZER_NAME
                        The name of this tokenizer
  --text-file TEXT_FILE
                        A text file to train the tokenizer on
```

You must specify the vocab size and tokenizer name in order to train a tokenizer, for example:

```
rye run train_tokenizer --vocab-size 1000 --name tokenizer-1k
```

By default, the text corpus that will be trained on is the tiny shakespeare dataset, which is located 
in `./tiny-shakespeare.txt`, but you can also specify your own text file.

## 3. Download A Dataset

Now, you will need to download a dataset.

You can do so with the command:

```text
rye run download_data <ds-name> [OPTIONS]
```

Currently the supported datasets are [`fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 
and [`smol`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

You are required to specify one of those dataset names.

Valid options for downloading data are:

```text
options:
  --ds-sample DS_SAMPLE
                        An optional sample of the dataset to download.
  -t TOKENIZER_PATH, --tokenizer-path TOKENIZER_PATH
                        The tokenizer to tokenize the data with
  --dir-prefix DATADIR_PREFIX
                        An optional prefix to add to the directory the data will be saved to
```

They are all optional. Note that if you skipped the previous step of training a tokenizer, 
the default tokenizer will be used, which is located in `tokenizers/base.json`.

So, a valid command would be:

`rye run download_data fineweb`

Or smollm-corpus:

`rye run download_data smol`

That's it for downloading data.

## 4. Train a model
> [!WARNING]
By default the tokenizer that's used for training is located in `tokenizers/base.json`. 
If the dataset you're using was downloaded with your own tokenizer, don't forget to use the 
same one for model training by specifying the `--tokenizer-path` option.

Finally, you can train your model.

Start the pre-training script like so:

```
rye run train_model [OPTIONS]
```

Here are the valid options:

```
options:
  -n MODEL_NAME, --model-name MODEL_NAME
  -d DATA_DIR, --data-dir DATA_DIR
  -t TOKENIZER_PATH, --tokenizer-path TOKENIZER_PATH
  -s SEED, --seed SEED
```

The model name and data dir are both required to be specified.

Thus, a valid command for model training would be:

`rye run train_model -n example_model --data-dir data/fineweb/sample-10BT`.


### Modifying Configuration

If you'd like to customize training further, you can do so by modifying the 
configuration files. 

To modify training configuration such as batch size, number of epochs, etc, 
edit the [training configuration class](./src/talm/config/training.py).

To modify model-specific configuration, like dimension sizes, edit the 
[model configuration class](./src/talm/config/model.py).

### SFT, RLHF

Right now, only pre-training a model is supported. In the future,
supervised fine tuning and RLHF may be implemented.

## Wrapping up

Nice, you just walked through training a language model with `talm`. Hope you enjoyed, 
the experience!
