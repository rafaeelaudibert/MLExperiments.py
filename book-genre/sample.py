# Libraries/Core modules
import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
from pprint import pprint
from keras.models import Model
from yattag import Doc, indent

# Custom code
from model import build_model, load_weights, build_model

# Default initialization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable TF debugging information

# Constants
DATA_DIR = './data'
MODEL_DIR = './model'
HTML_DIR = './html'
TRUNCATE_SIZE = 64
FILL_CARACHTER = '|'
FILLING = FILL_CARACHTER * TRUNCATE_SIZE

# Print a value based on a conditional
def print_debug(message, condition):
    if condition:
        pprint(message)

# Generate classes for the HTML document
def generate_class(normalized_value):
    if normalized_value < 0.10:
        return "_10"
    elif normalized_value < 0.20:
        return "_20"
    elif normalized_value < 0.30:
        return "_30"
    elif normalized_value < 0.40:
        return "_40"
    elif normalized_value < 0.50:
        return "_50"
    elif normalized_value < 0.60:
        return "_60"
    elif normalized_value < 0.70:
        return "_70"
    elif normalized_value < 0.80:
        return "_80"
    elif normalized_value < 0.90:
        return "_90"
    else:
        return "_100"

def sample(epoch, seed, generateHTML=False, debug=False):

    # Configure the seed with the filling
    seed = (seed + FILLING)[:TRUNCATE_SIZE]
    
    # Fetch the file with the genres dataset
    print_debug('Fetching genres.json file', debug)
    with open(os.path.join(DATA_DIR, 'genres.json'), 'r') as f:
        genre_to_idx = json.load(f)
        idx_to_genre = { i: ch for (ch, i) in list(genre_to_idx.items()) }    
        output_size = len(idx_to_genre)

    # Fetch the file with the char-to-index dataset
    print_debug('Fetching char_to_idx.json file', debug)
    with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'r') as f:
            char_to_idx = json.load(f)
            idx_to_char = {i: ch for (i, ch) in enumerate(char_to_idx)}
            vocab_size = len(idx_to_char)
        

    # Build the normal model
    print_debug('Loading model from file', debug)
    model = build_model(1, 1, vocab_size, output_size)
    load_weights(epoch, model)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

    # Predict the result
    print_debug('Running results prediction', debug)
    activations = []
    for c in seed:
        batch = np.zeros((1, 1))
        batch[0, 0] = char_to_idx[c]
        result = [(idx_to_genre[i], prob) for (i, prob) in enumerate(model.predict_on_batch(batch).ravel())]
     
    # HTML Generation
    if generateHTML:
        print_debug('Starting HTML generation', not debug)

        # Build the model with all the outputs for the HTML generation
        print_debug('Loading new model from file with all output layers', debug)
        layer_dict = {layer.name: layer for layer in model.layers}
        outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=outputs)
        load_weights(epoch, activation_model)

        if not os.path.exists(HTML_DIR):
            os.makedirs(HTML_DIR)

        # Proper HTML generation
        for layer_name in layer_dict:            
            doc, tag, _, line = Doc().ttl()       # Get HTML creation variables

            print_debug('Starting to create {} HTML file'.format(layer_name), debug)
            with open('{}/{}.html'.format(HTML_DIR, layer_name), 'w') as f:
                with tag('html'):
                    doc.asis('<link rel="stylesheet" href="main.css">')
                    doc.asis('<link href="https://fonts.googleapis.com/css?family=Muli" rel="stylesheet">')
                    with tag('body'):
                        # Each neuron
                        for i in range(layer_dict[layer_name].output.shape[-1]):
                            print_debug('Iterating neuron {}'.format(i), debug)
                            with tag('p'):  
                                line('span', 'Neuron {:03d} - '.format(i + 1))
                                for c in seed:
                                    batch = np.zeros((1, 1))
                                    batch[0, 0] = char_to_idx[c]
                                    activations.append({list(layer_dict.keys())[i]: activation for (i, activation) in enumerate(activation_model.predict_on_batch(batch))})            
                                    
                                    # Fetching neuron and normalizing it value in the klass
                                    neuron = activations[-1][layer_name].squeeze()[i]   
                                    line('span', c, klass=generate_class(neuron * 0.5 + 0.5))
                    
                # Write HTML to the file
                f.write(doc.getvalue())
                print_debug('{} HTML file created'.format(layer_name), debug)
    
    # Sort the result and returns it
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)    
    print_debug(sorted_result, debug)
    return sorted_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    parser.add_argument('--epoch', type=int, default=150, help='epoch checkpoint to sample from (default 150)')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    parser.add_argument('--html', dest='html', action='store_true')
    parser.add_argument('--pprint', dest='pprint', action='store_true')
    parser.set_defaults(html=False, pprint=False)
    args = parser.parse_args()

    result = sample(args.epoch, args.seed, args.html, args.pprint)
    print("This book belongs to the {} category with {:02.2f}% confidence".format(result[0][0], result[0][1] * 100))
    for extra_results in result[1:]:
        if extra_results[1] > 0.10:
            print("It might also belongs to the {} category with {:02.2f}% confidence".format(extra_results[0], extra_results[1] * 100))
        pass
    
