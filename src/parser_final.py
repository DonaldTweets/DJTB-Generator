import numpy as np
import re
"""
Fichier de test, inutile pour le moment
"""


OCC_LIMIT = 10


def parser_fichier(filepath):
    print("Starting Data preparation...\n")
    
    # Lecture et caractérisation du corpus
    text = open(filepath, 'r').read().lower()
    chars = list(set(text))
    vocab_size = len(chars)
    
    # Suppression de certains caractères spéciaux polluant la compréhension de la machine
    re.sub(r"\n", ' ', text)
    
    # Détection des caractères n'apparaissant pas au moins OCC_LIMIT fois dans le corpus
    nb_occ_chars = np.zeros(len(chars))
    for i in range(len(chars)):
        for j in range(len(text)):
            if text[j] == chars[i]:
                nb_occ_chars[i] += 1
    
    vocab_occ = dict(zip(chars, nb_occ_chars))
    key_blacklist = []
    for key in vocab_occ:
        if vocab_occ[key] < OCC_LIMIT:
            key_blacklist.append(key)
    
    print("Caracteres supprimes :\n", key_blacklist, "\n(Insufficient occurences in corpus)\n")
    for k in key_blacklist:
        re.sub(k, " ", text)
        del vocab_occ[k]
    
    print(text)
        
    return vocab_size, vocab_occ


if __name__ == "__main__":
    vocab_s, vocab_o = parser_fichier("./data/tweets_small_raw.txt")
    print (vocab_s, "\n", vocab_o)
