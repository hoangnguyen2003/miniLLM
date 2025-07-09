import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line.strip()])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    return tokenizer, max_sequence_len, input_sequences