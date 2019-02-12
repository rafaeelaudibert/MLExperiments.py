import argparse
import json
import os
from pathlib import Path
import sys
import csv
import random

import numpy as np

from model import build_model, save_weights, load_weights

DATA_DIR = './data'
LOG_DIR = './logs'
MODEL_DIR = './model'

BATCH_SIZE = 32
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file, resume=0):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = resume
        if not resume:
            with open(self.file, 'w') as f:
                f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)


def read_batches(I, O, vocab_size, output_size):
    batch_chars = I.shape[0] // BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, output_size))
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = I[batch_chars * batch_idx + start + i]
            Y[batch_idx, int(O[(batch_chars * batch_idx + start) // SEQ_LENGTH])] = 1
        yield X, Y

def train(raw_in_data, raw_out_data, genres, epochs, save_freq, resume=False):

    in_data, out_data = zip(*[(in_data_value, raw_out_data[index]) for (index, in_data_value) in enumerate(raw_in_data) if random.random() < 0.70])

    text = ''.join(in_data)

    if resume:
        print("Attempting to resume last training...")

        model_dir = Path(MODEL_DIR)
        c2ifile = model_dir.joinpath('char_to_idx.json')
        with c2ifile.open('r') as f:
            char_to_idx = json.load(f)

        checkpoints = list(model_dir.glob('weights.*.h5'))
        if not checkpoints:
            raise ValueError("No checkpoints found to resume from")

        resume_epoch = max(int(checkpoint.name.split('.')[1]) for checkpoint in checkpoints)
        print("Resuming from epoch", (epochs - resume_epoch))
    else:
        resume_epoch = 0
        char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
        with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'w') as f:
            json.dump(char_to_idx, f)

    vocab_size = len(char_to_idx)
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size, len(genres))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if resume:
        load_weights(resume_epoch, model)

    with open('data/genres.json', 'r') as json_file:
        genres = json.load(json_file)

    I = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    log = TrainLogger('training_log.csv', resume_epoch)

    batch_quantity = ((I.shape[0] // BATCH_SIZE) - SEQ_LENGTH) / SEQ_LENGTH

    for epoch in range(resume_epoch, epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        losses, accs = [], []
        for i, (X, Y) in enumerate(read_batches(I, out_data, vocab_size, len(genres))):
            batch_percentage = (i + 1) / batch_quantity
            total_percentage = ((i + 1) + (epoch - resume_epoch) * batch_quantity) / (batch_quantity * (epochs - resume_epoch + epoch))

            loss, acc = model.train_on_batch(X, Y)
            print("Batch {} (B: {:.6f} | T: {:.6f}): loss = {:.6f}, acc = {:.6f}".format(i + 1, batch_percentage, total_percentage, loss, acc))
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print("Saved checkpoint to weights.{}.h5".format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='book-data.csv',
                        help='name of the csv file to train from')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10,
                        help='checkpoint save frequency')
    parser.add_argument('--resume', action='store_true',
                        help='resume from previously interrupted training')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Reading book data
    X, Y = [], []
    with open('data/parsed-book-data.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            X.append(row[0])
            Y.append(row[1])

    with open('data/genres.json') as json_file:
        genres = json.load(json_file)

    train(X, Y, genres, args.epochs, args.freq, args.resume)
