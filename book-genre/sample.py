import argparse
import json
import os
from pathlib import Path

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

from model import build_model, load_weights

DATA_DIR = './data'
MODEL_DIR = './model'
TRUNCATE_SIZE = 64
FILL_CARACHTER = '|'
FILLING = FILL_CARACHTER * TRUNCATE_SIZE

def build_sample_model(vocab_size, output_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, batch_input_shape=(1, 1)))
    for i in range(2):
        model.add(LSTM(128, return_sequences=(i != 1), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    return model

def sample(epoch, header):

    header = (header + FILLING)[:TRUNCATE_SIZE]
    
    with open(os.path.join(DATA_DIR, 'genres.json'), 'r') as f:
        genre_to_idx = json.load(f)
        idx_to_genre = { i: ch for (ch, i) in list(genre_to_idx.items()) }
    

    c2ifile = Path(MODEL_DIR).joinpath('char_to_idx.json')
    with c2ifile.open('r') as f:
            char_to_idx = json.load(f)
            idx_to_char = {i: ch for (i, ch) in enumerate(char_to_idx)}
    
    vocab_size = len(idx_to_char)
    output_size = len(idx_to_genre)

    model = build_sample_model(vocab_size, output_size)
    load_weights(epoch, model)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))
   

    # Iterating through all the carachters, except the last one,
    # to make the RNN to "remember" about the seed text
    for c in header:
        batch = np.zeros((1, 1))
        batch[0, 0] = char_to_idx[c]
        result = model.predict_on_batch(batch).ravel()

    print({idx_to_genre[i]: prob for (i, prob) in sorted(enumerate(result), key=lambda x: x[1], reverse=True)})
    ordered = [idx_to_genre[i] for (i, prob) in sorted(enumerate(result), key=lambda x: x[1], reverse=True)]
    
    return ordered[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    parser.add_argument('--epoch', type=int, default=150, help='epoch checkpoint to sample from (default 100)')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    args = parser.parse_args()

    category = sample(args.epoch, args.seed)
    print(category)
