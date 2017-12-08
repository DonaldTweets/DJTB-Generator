from __future__ import print_function
import numpy as np


# Chargement et preparation du texte brut
def load_data(data_dir, seq_length, verbose=True):
    if verbose:
        print("Chargement du corpus...\n")
    data = open(data_dir, 'r').read()
    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    if verbose:
        print("Longueur du corpus: ", len(data), " caracteres.")
        print("Vocabulaire total: ", VOCAB_SIZE, " caracteres")

    # Dictionnaire liant chaque caractere a un entier (necessaire pour que le reseau comprenne !)
    ix_to_char = {ix: char for ix, char in enumerate(chars)}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}

    # Le // evite de placer un float dans un in range...
    x = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))
    y = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))
    
    for i in range(0, len(data)//seq_length):
        x_sequence = data[i*seq_length:(i+1)*seq_length]
        x_sequence_ix = [char_to_ix[value] for value in x_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        
        for j in range(seq_length):
            input_sequence[j][x_sequence_ix[j]] = 1.
            x[i] = input_sequence

        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
            
    return x, y, VOCAB_SIZE, ix_to_char


# Generation d'un texte
def generate_text(model, length, vocab_size, ix_to_char):
    
    # On donne un seed (en la forme d'un caractere choisi aleatoirement)
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    x = np.zeros((1, length, vocab_size))
    
    for i in range(length):
        # On ajoute le caractere predit a la sequence
        x[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(x[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)
