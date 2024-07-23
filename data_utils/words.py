
import gensim.downloader

def load_words(path):
    with open(path, 'r') as f:
        words = f.read().splitlines()
    return words

def retrieve_word_lists(noun_path, vocab_path):
    nouns = load_words(noun_path)
    vocab = load_words(vocab_path)
    return nouns, vocab


def load_word2vec_model(model_name):
    return gensim.downloader.load(model_name)