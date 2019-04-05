import pickle
import sys
import gensim
import numpy as np


# prepare word to index dict and embedding matrix

def prepare_matrices(dict_file, embedding_file):

    m = 'word2vec.txt'  # file with used word embedding model
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)  # load word embedding model

    all_words = set()
    with open('./Chunk/train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 1:
                all_words.add(line.split('\t')[0])
    with open('./Chunk/dev.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 1:
                all_words.add(line.split('\t')[0])
    with open('./Chunk/test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 1:
                all_words.add(line.split('\t')[0])

    all_words = list(all_words)
    word_dict = set()

    # words both in files and in embedding model
    for word in all_words:
        if word in model:
            word_dict.add(word)

    word_dict = list(word_dict)
    word2idx = {w: i + 1 for i, w in enumerate(word_dict)}  # dict of word:index

    word_emb_len = len(model.get_vector(','))  # length of word embedding vector feature

    # prepare embedding matrix
    embedding_matrix = np.zeros((len(word_dict) + 1, word_emb_len))
    for word, i in word2idx.items():
        embedding_matrix[i] = model.get_vector(word)

    # save dict
    with open(dict_file, 'wb') as df:
        pickle.dump(word2idx, df)

    # save embedding matrix
    with open(embedding_file, 'wb') as ef:
        pickle.dump(embedding_matrix, ef)


if __name__ == '__main__':
    dict_file_name = sys.argv[1]
    embedding_file_name = sys.argv[2]

    prepare_matrices(dict_file_name, embedding_file_name)
