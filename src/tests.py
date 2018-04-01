from functions_DJTB import load_and_parse, format_data

chars, rawtext = load_and_parse('./data/tweets_small_raw.txt', verbose=True, pad_to_tweets=False)
x1, y1, vocab_size1, ix_to_char1 = format_data(chars, rawtext, 280, verbose_x=True)
