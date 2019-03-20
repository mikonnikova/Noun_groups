import gensim
import numpy as np
import pickle
import os.path
import sys
from prepare_matrice import prepare_matrices


# dictionary of part-of-speech tags
pos_dict = {'ADJ': 1, 'ADV': 2, 'INTJ': 3, 'NOUN': 4, 'PROPN': 5, 'VERB': 6, 'ADP': 7, 'AUX': 8, 'CCONJ': 9, 'DET': 10,
            'NUM': 11, 'PART': 12, 'PRON': 13, 'SCONJ': 14, 'PUNCT': 15, 'SYM': 16, 'X': 17}

# variants of additional part-of-speech features
feats_variants = {'Foreign': {'Yes': 1},
                  'Gender': {'Fem': 1, 'Masc': 2, 'Neut': 3},
                  'Animacy': {'Anim': 1, 'Inan': 2},
                  'Number': {'Plur': 1, 'Sing': 2},
                  'Case': {'Acc': 1, 'Nom': 2, 'Dat': 3, 'Gen': 4, 'Ins': 5, 'Loc': 6, 'Par': 7, 'Voc': 8},
                  'VerbForm': {'Conv': 1, 'Fin': 2, 'Inf': 3, 'Part': 4},
                  'Mood': {'Ind': 1, 'Imp': 2, 'Cnd': 3},
                  'Tense': {'Fut': 1, 'Past': 2, 'Pres': 3},
                  'Person': {'1': 1, '2': 2, '3': 3},
                  'Aspect': {'Imp': 1, 'Perf': 2},
                  'Voice': {'Act': 1, 'Mid': 2, 'Pass': 3}}


# transform word information into a vector representation

def process_sentence(words, pos, feats, iob_tags, word2idx):

    # get word embedding from a model
    word_vector = [word2idx[w] if w in word_dict else 0 for w in words]

    # get part-of-speech feature
    pos_vector = [pos_dict[p] if p in pos_dict else 0 for p in pos]

    # get part-of-speech additional features (array of arrays)
    global_feats_vector = np.zeros((len(feats), len(feats_list)))

    for i in range(len(feats)):
        local_feats_list = [feat.split('=') for feat in feats[i].split('|')]
        feats_dict = {}
        for feat in local_feats_list:
            if len(feat) > 1:
                feats_dict[feat[0]] = feat[1]

        for j in range(len(feats_list)):
            if feats_list[j] in feats_dict:
                global_feats_vector[i][j] = feats_variants[feats_list[j]][feats_dict[feats_list[j]]]

    # add IOB2 tag
    tag = np.zeros((len(iob_tags), 3))
    for i in range(len(iob_tags)):
        if iob_tags[i] == 'B':
            tag[i][0] = 1
        elif iob_tags[i] == 'I':
            tag[i][1] = 1
        elif iob_tags[i] == 'O':
            tag[i][2] = 1

    return word_vector, pos_vector, global_feats_vector, tag


# process corpus in IOB2 format
# return pickle file with word features for all sentences

def process_corpus(corpus_file, words_file, pos_file, feats_file, answers_file, dict_file):

    if not os.path.exists(dict_file):
        prepare_matrices(dict_file, 'embedding_matrix.pkl')
    with open(dict_file, 'rb') as df:
        word2idx = pickle.load(df)

    with open(corpus_file, 'r', encoding='utf-8') as in_f:

        words, pos, feats, tags = [], [], [], []  # list for a sentence
        words_vector, pos_vector, feats_vector, answer = [], [], [], []  # global array

        for line in in_f:
            if len(line) <= 1:  # empty line
                if len(words) > 0:
                    info1, info2, info3, tags = process_sentence(words, pos, feats, tags, word2idx)
                    words_vector.append(info1)
                    pos_vector.append(info2)
                    feats_vector.append(info3)
                    answer.append(tags)
                words, pos, feats, tags = [], [], [], []
                continue

            if line[0] == '#':  # info line
                continue

            parsed_line = line.split('\t')
            words.append(parsed_line[0])
            pos.append(parsed_line[1])
            feats.append(parsed_line[2])
            tags.append(parsed_line[3][:-1])

        if len(words) > 0:
            info1, info2, info3, tags = process_sentence(words, pos, feats, tags, word2idx)
            words_vector.append(info1)
            pos_vector.append(info2)
            feats_vector.append(info3)
            answer.append(tags)

    with open(words_file, 'wb') as wf:
        pickle.dump(words_vector, wf)
    with open(pos_file, 'wb') as pf:
        pickle.dump(pos_vector, pf)
    with open(feats_file, 'wb') as ff:
        pickle.dump(feats_vector, ff)
    with open(answers_file, 'wb') as af:
        pickle.dump(answer, af)

    return


if __name__ == '__main__':
    corpus_file_name = sys.argv[1]  # name of corpus to process
    words_file_name = sys.argv[2]  # where to save words features
    pos_file_name = sys.argv[3]  # where to save pos features
    feats_file_name = sys.argv[4]  # where to save feats features
    answers_file_name = sys.argv[5]  # where to save answers
    dict_file_name = sys.argv[6]  # word:index dict file
    process_corpus(corpus_file_name, words_file_name, pos_file_name, feats_file_name, answers_file_name, dict_file_name)
