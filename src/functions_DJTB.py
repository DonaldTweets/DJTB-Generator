import numpy as np
import re
"""

"""


OCC_LIMIT = 10


def load_and_parse(filepath, verbose=True, pad_to_tweets=False, tweet_length=280):
    
    if verbose:
        print("Starting Data parsing...\n")
    
    # Lecture et caracterisation du corpus
    text = open(filepath, 'r').read().lower()
    charset = list(set(text))
    vocab_size = len(charset)
    
    # Suppression de certains caractères speciaux polluant la comprehension de la machine
    re.sub(r"\n", ' ', text)
    
    # Détection des caractères n'apparaissant pas au moins OCC_LIMIT fois dans le corpus
    nb_occ_chars = np.zeros(len(charset))
    for i in range(len(charset)):
        for j in range(len(text)):
            if text[j] == charset[i]:
                nb_occ_chars[i] += 1
    
    vocab_occ = dict(zip(charset, nb_occ_chars))
    key_blacklist = []
    for key in vocab_occ:
        if vocab_occ[key] < OCC_LIMIT:
            key_blacklist.append(key)

    # La suppression des caractères trop peu nombreux dans le corpus prend en compte les caracteres speciaux
    # et s'efforce de les rendre lisibles dans les regular expressions en ajoutant un antislash
    unreadable_chars = ['|', '.', '*' '^', '$', '+', '?']
    
    for k in key_blacklist:
        if k in unreadable_chars:
            readable_k = '\\' + k
        else:
            readable_k = k
            
        text = re.sub(readable_k, '', text)
        
        del vocab_occ[k]
    print("Deleted following characters :\n", key_blacklist, "\n(Insufficient occurences in corpus)\n")
    
    # Suppression des 'http://www. ' qui ne menent à rien et ajout d'espace avant les liens n'en ayant pas
    text = re.sub('([0-9]|[a-z]|:|!)(http://|https://)', '\g<1> \g<2>', text)
    text = re.sub('(http://www.|https://www.|http://)\n', '', text)
    # Suppression des doubles et triples espaces
    text = re.sub(' +', ' ', text)
    
    if pad_to_tweets:
        print("Padding tweets...")
        iterator = 0
        old_iterator = 0
        text = text + '£'
        
        while text[iterator] != '£':
            if text[iterator] == '\n' and text[iterator + 1] != '£':
                padding_string = " " * (tweet_length - (iterator - old_iterator))
                
                text = text[:iterator] + padding_string + text[(iterator+1):]
                old_iterator += tweet_length
                iterator += len(padding_string)
            
            iterator += 1
    
    return charset, text


def format_data(charset, data, sequence_length):
    """
    
    :param sequence_length:
    :param charset: set contenant tous les caracteres utilises par le texte
    :param data: texte brut pre-nettoye (à l'aide de load_and_parse)
    :return:
    """
    
    # Dictionnaire liant chaque caractere a un entier et vice-versa(necessaire pour que le reseau les comprenne !)
    ix_to_char = {ix: char for ix, char in enumerate(charset)}
    char_to_ix = {char: ix for ix, char in enumerate(charset)}
    
    vocab_size = len(charset)

    # Creation de matrices de donnees. On va en fait decouper ensuite nos donnees en sequences de caracteres de longueur
    # sequence_length. La matrice de donnees en 3 dimensions : une ligne correspond a une sequence, une colonne a un
    # caractere dans cette sequence
    # Le // evite de placer un float dans un in range. Je doute de la proprete mais jusqu'ici pas de soucis
    x = np.zeros((len(data) // sequence_length, sequence_length, vocab_size))
    y = np.zeros((len(data) // sequence_length, sequence_length, vocab_size))
    
    # Le gros du boulot. Remplissage de la matrice ligne par ligne.
    for i in range(0, len(data) // sequence_length):
        x_sequence = data[i * sequence_length:(i + 1) * sequence_length]
        x_sequence_ix = [char_to_ix[value] for value in x_sequence]
        
        input_sequence = np.zeros((sequence_length, vocab_size))
    
        for j in range(sequence_length):
            input_sequence[j][x_sequence_ix[j]] = 1.
            x[i] = input_sequence
    
        y_sequence = data[i * sequence_length + 1:(i + 1) * sequence_length + 1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((sequence_length, vocab_size))
    
        for j in range(sequence_length) :
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence

    return x, y, vocab_size, ix_to_char


# Generation d'un texte utilisant un modele existant
def generate_text(model, length, vocab_size, ix_to_char):
    # On donne un seed (en la forme d'un caractere choisi aleatoirement)
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    x = np.zeros((1, length, vocab_size))
    
    for i in range(length):
        # On ajoute le caractere predit a la sequence
        x[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end = "")
        ix = np.argmax(model.predict(x[:, :i + 1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


# --------------------------------TESTING------------------------------
if __name__ == "__main__":
    chars, txt = load_and_parse("./data/tweets_small_raw.txt", pad_to_tweets = True)
    x, y, v_s, tochar = format_data(chars, txt, 280)
