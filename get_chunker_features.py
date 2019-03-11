import gensim
import numpy as np
import pickle
import sys


m = 'word2vec.txt'  # file with used word embedding model
model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)  # load word embedding model

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

# list of possible features
feats_list = [feat for feat in feats_variants]

word_emb_len = len(model.get_vector(',')) + 1  # length of word embedding vector feature
pos_len = len(pos_dict) + 1  # length of part-of-speech vector feature
feats_len = 0  # length of additional part-of-speech features vector feature
for f in feats_list:
    feats_len += (len(feats_variants[f]) + 1)
all_len = word_emb_len + pos_len + feats_len  # total length of a features vector for a single word


# transform word information into a vector representation

def get_features(word, pos, feats, iob_tag):
    # get word embedding from a model
    word_vector = np.concatenate(([1], model.get_vector(word)), axis=None) \
        if word in model else np.zeros(word_emb_len)

    # add part-of-speech feature one-hot encoding
    pos_vector = np.zeros(pos_len)
    if pos in pos_dict:
        pos_vector[pos_dict[pos]] = 1
        pos_vector[0] = 1

    # add part-of-speech additional features one-hot encodings
    local_feats_list = [feat.split('=') for feat in feats.split('|')]
    feats_dict = {}
    for feat in local_feats_list:
        if len(feat) > 1:
            feats_dict[feat[0]] = feat[1]
    feats_vector = np.zeros(0)
    for feat in feats_list:
        local = np.zeros(len(feats_variants[feat]) + 1)
        if feat in feats_dict:
            local[0] = 1
            local[feats_variants[feat][feats_dict[feat]]] = 1
        feats_vector = np.append(feats_vector, local)

    # add IOB2 tag
    tag = np.zeros(1)
    if iob_tag == 'B':
        tag[0] = 1
    elif iob_tag == 'I':
        tag[0] = 2

    return np.concatenate((word_vector, pos_vector, feats_vector), axis=None), tag


# for a given sentence get all words features

def process_sentence(info):
    sentence_features = []
    tags = []
    info_items = info.split('\t')

    for word_count in range(len(info_items) // 4):
        word_features, tag = get_features(info_items[word_count * 4], info_items[word_count * 4 + 1],
                                          info_items[word_count * 4 + 2], info_items[word_count * 4 + 3])

        sentence_features.append(word_features)
        tags.append(tag)

    return np.array(sentence_features), np.array(tags)


# pad or cut the sentence to the length of 70

def sentence_vector(words, tags):
    if len(words) > 70:
        return words[:70], tags[:70]

    filler = np.zeros((70 - len(words), all_len))
    sentence = np.concatenate((words, filler), axis=0)

    filler = np.zeros((70 - len(tags), 1))
    answer = np.concatenate((tags, filler), axis=0)

    return sentence, answer


# process corpus in IOB2 format
# return pickle file with word features for all sentences

def process_corpus(corpus_file, features_file, answers_file):
    with open(corpus_file, 'r', encoding='utf-8') as in_f, open(features_file, 'wb') as res_file, \
            open(answers_file, 'wb') as ans_file:
        info = ''
        res = np.zeros((70, all_len))
        answer = np.zeros((70, 1))

        # process file line by line
        for line in in_f:
            if len(line) <= 1:
                if len(info) > 0:
                    words, tags = process_sentence(info)
                    sent, tags = sentence_vector(words, tags)
                    res = np.dstack((res, sent))
                    answer = np.dstack((answer, tags))
                info = ''
                continue

            parsed_line = line.split(maxsplit=3)
            if parsed_line[0] == '#':
                continue

            info += (line[:-1] + '\t')

        if len(info) > 0:
            words, tags = process_sentence(info)
            sent, tags = sentence_vector(words, tags)
            res = np.dstack((res, sent))
            answer = np.dstack((answer, tags))

        # rearrange the axes
        res = np.moveaxis(res, 1, 2)
        res = np.moveaxis(res, 0, 1)
        res = res[1:]

        answer = np.moveaxis(answer, 1, 2)
        answer = np.moveaxis(answer, 0, 1)
        answer = answer[1:]

        # save to files
        pickle.dump(res, res_file)
        pickle.dump(answer, ans_file)

    return


if __name__ == '__main__':
    corpus_file_name = sys.argv[1]
    features_file_name = sys.argv[2]
    answers_file_name = sys.argv[3]
    process_corpus(corpus_file_name, features_file_name, answers_file_name)
