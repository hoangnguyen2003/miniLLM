def tokenize(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]