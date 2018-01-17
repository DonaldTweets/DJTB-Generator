from __future__ import print_function

from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

from functions_DJTB import *

# Parsing d'arguments console => definition des parametres du reseau de neurones
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/tweets_small_raw.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=280)
ap.add_argument('-hidden_dim', type=int, default=600)
ap.add_argument('-generate_length', type=int, default=280)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
ap.add_argument('-model', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
MODEL = args['model']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Chargement du corpus d'entrainement
chars, rawtext = load_and_parse(DATA_DIR, verbose=True, pad_to_tweets=True)

X, y, VOCAB_SIZE, ix_to_char = format_data(chars, rawtext, SEQ_LENGTH)

# Creation du reseau...
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

if not WEIGHTS == '':
    if not MODEL == '':
        # On ne fait que la generation
        # load du modele
        yaml_file = open(MODEL, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(WEIGHTS)
        print("Loaded model...")
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
        
    else:
        model.load_weights(WEIGHTS)
        nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
    nb_epoch = 0

# Si on ne precise pas de poids => on entraine le reseau from scratch
if args['mode'] == 'train' or WEIGHTS == '':
    while True:
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
        nb_epoch += 1
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
        
        if nb_epoch % 5 == 0:
            # Serialisation du modele
            model_yaml = model.to_yaml()
            with open("model_{}.yaml".format(HIDDEN_DIM), "w") as yaml_file :
                yaml_file.write(model_yaml)
            # Serialisation des poids
            model.save_weights("model_{}_hidden_{}.h5".format(nb_epoch, HIDDEN_DIM))
            print("Model saved...")

# Sinon, on genere a partir du modele
elif WEIGHTS == '':
    model.load_weights(WEIGHTS)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    print('\n\n')
else:
    print('\n\nNothing to do!')