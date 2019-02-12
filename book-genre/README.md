# book-genre

Multi-layer recurrent neural networks for detecting a book category, inspired by [Andrej Karpathy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness) and the original torch source code [karpathy/char-rnn](https://github.com/karpathy/char-rnn), as well as [Eric Zhang char-rnn-keras implementation](https://github.com/ekzhang/char-rnn-keras).

The dataset is the same used on [this paper](https://arxiv.org/abs/1610.09204). It can be found and downloaded [here](https://github.com/uchidalab/book-dataset/blob/master/Task2).

## Requirements

This code is written in Python 3, and it requires the [Keras](https://keras.io) deep learning library.

## Input data

All input data should be placed in the [`data/`](./data) directory.

## Usage

It is necessary to parse the data to a format which can be used by the training, running:
```bash
$ python3 sample.py
```

To train the model with default settings:
```bash
$ python3 train.py
```

To sample the model (epoch 100 as default):
```bash
$ python3 sample.py
```

Training loss/accuracy is stored in `logs/training_log.csv`. Model results, including intermediate model weights during training, are stored in the `model` directory. These are also used by `sample.py` for sampling.
