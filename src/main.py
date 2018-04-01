from __future__ import print_function

from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import numpy as np
import argparse

from functions_DJTB import *

# Parsing d'arguments console => definition des parametres du reseau de neurones
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='../data/tweets_small_raw.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=280)
ap.add_argument('-hidden_dim', type=int, default=600)
ap.add_argument('-generate_length', type=int, default=280)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
ap.add_argument('-model', default='')
ap.add_argument('-padding', default=False)
ap.add_argument('-save_to_file', default=False)
args = vars(ap.parse_args())

arg_data_dir = args['data_dir']
arg_batch_size = args['batch_size']
arg_hidden_dim = args['hidden_dim']
arg_seq_lgth = args['seq_length']
arg_weights = args['weights']
arg_model = args['model']
arg_padding = args['padding']
arg_save_to_file = args['save_to_file']

arg_generate_length = args['generate_length']
arg_layer_num = args['layer_num']

id_generation = np.random.randint(10000)

# Chargement du corpus d'entrainement
chars, rawtext = load_and_parse(arg_data_dir, verbose=True, pad_to_tweets=arg_padding)

X, y, vocab_size, ix_to_char = format_data(chars, rawtext, arg_seq_lgth)

# Creation du reseau...
model = Sequential()
model.add(LSTM(arg_hidden_dim, input_shape=(None, vocab_size), return_sequences=True))
for i in range(arg_layer_num - 1):
    model.add(LSTM(arg_hidden_dim, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

if not arg_weights == '':
    if not arg_model == '':
        # On ne fait que la generation
        # load du modele
        yaml_file = open(arg_model, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(arg_weights)
        print("Loaded model...")
        generate_text(model, arg_generate_length, vocab_size, ix_to_char, arg_save_to_file)
        
    else:
        model.load_weights(arg_weights)
        nb_epoch = int(arg_weights[arg_weights.rfind('_') + 1:arg_weights.find('.')])
else:
    nb_epoch = 0

# Si on ne precise pas de poids => on entraine le reseau from scratch
if args['mode'] == 'train' or arg_weights == '':
    while True:
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(X, y, batch_size=arg_batch_size, verbose=1, nb_epoch=1)
        nb_epoch += 1
        generate_text(model, arg_generate_length, vocab_size, ix_to_char, number=10, save_to_file=arg_save_to_file,
                      seed=str(id_generation)+str(nb_epoch))
        
        if nb_epoch % 5 == 0:
            # Serialisation du modele
            model_yaml = model.to_yaml()
            with open("model_{}.yaml".format(arg_hidden_dim), "w") as yaml_file :
                yaml_file.write(model_yaml)
            # Serialisation des poids
            model.save_weights("model_{}_hidden_{}.h5".format(nb_epoch, arg_hidden_dim))
            print("Model saved...")

# Sinon, on genere a partir du modele
elif arg_weights == '':
    model.load_weights(arg_weights)
    generate_text(model, arg_generate_length, vocab_size, ix_to_char)
    print('\n\n')
else:
    print('\n\nNothing to do!')