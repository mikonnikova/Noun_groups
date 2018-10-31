import numpy as np
from keras.models import load_model
import os
import pickle
import gensim


clf = load_model("nn.h5")
	
m = 'ruscorpora_upos_skipgram_300_5_2018.vec'  # file with used word embedding model
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

word_emb_len = len(model.get_vector('студент_NOUN')) + 1  # length of word embedding vector feature
pos_len = len(pos_dict) + 1  # length of part-of-speech vector feature
feats_len = 0  # length of additional part-of-speech features vector feature
for f in feats_list:
    feats_len += (len(feats_variants[f]) + 1)
all_len = word_emb_len + pos_len + feats_len + 3  # total length of a features vector for a single word

def transform_corpus(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        text = ''
        for line in in_f:
            if len(line) <= 1:  # go to the next text feature
                if len(text) > 0:
                    print(text, file=out_f)
                    text = ''
                print(file=out_f)
                continue
            parsed_line = line.split()
            if parsed_line[0] == '#':
                if parsed_line[1] == 'text':  # get given sentence
                    print(line[:-1], file=out_f)
                continue
            text = text + parsed_line[0] + ' ' + parsed_line[1] + ' ' + parsed_line[2] + ' ' + parsed_line[3] + ' ' \
                   + parsed_line[5] + ' '  # add word information
				   
def get_info_from_corpus(corpus_info):
    words = []
    pos = []
    feats = []
    entities = []

    info = corpus_info.split()
    for word_count in range(len(info) // 5):
        words.append(info[word_count * 5 + 2])  # lemmas
        pos.append(info[word_count * 5 + 3])  # part-of-speech tags
        feats.append(info[word_count * 5 + 4])  # additional part-of-speech features
        # get words starting with title letters (entities candidates)
        if word_count > 0 and info[word_count * 5 + 1].istitle():
            entities.append(1)
        else:
            entities.append(0)

    return words, pos, feats, entities
	
def get_word_features(num, words, pos, feats, entities, root_num):
    word_pos = words[num] + '_' + pos[num]
    # get word embedding from a model
    word_vector = np.concatenate(([1], model.get_vector(word_pos)), axis=None) \
        if word_pos in model else np.zeros(word_emb_len)

    # add part-of-speech feature one-hot encoding
    pos_vector = np.zeros(pos_len)
    if pos[num] in pos_dict:
        pos_vector[pos_dict[pos[num]]] = 1
        pos_vector[0] = 1

    # add part-of-speech additional features one-hot encodings
    local_feats_list = [feat.split('=') for feat in feats[num].split('|')]
    feats_dict = {}
    for feat in local_feats_list:
        if len(feat) > 1:
            feats_dict[f[0]] = feat[1]
    feats_vector = np.zeros(0)
    for feat in feats_list:
        local = np.zeros(len(feats_variants[feat]) + 1)
        if feat in feats_dict:
            local[0] = 1
            local[feats_variants[feat][feats_dict[feat]]] = 1
        feats_vector = np.append(feats_vector, local)

    if root_num == -1:
        return np.concatenate((word_vector, pos_vector, feats_vector, entities[num]), axis=None)
    offset = abs(num - root_num)
    direction = (num - root_num) / offset if root_num != num else 0

    return np.concatenate(([1], word_vector, pos_vector, feats_vector, entities[num], offset, direction), axis=None)
	
def get_features(num, words, pos, feats, entities, root_num):
    if num >= 2:
        pre_pre_word = get_word_features(num - 2, words, pos, feats, entities, root_num)
    else:
        pre_pre_word = np.zeros(all_len + 1)
    if num >= 1:
        pre_word = get_word_features(num - 1, words, pos, feats, entities, root_num)
    else:
        pre_word = np.zeros(all_len + 1)
    word = get_word_features(num, words, pos, feats, entities, root_num)
    if num <= len(words) - 2:
        post_word = get_word_features(num + 1, words, pos, feats, entities, root_num)
    else:
        post_word = np.zeros(all_len + 1)
    if num <= len(words) - 3:
        post_post_word = get_word_features(num + 2, words, pos, feats, entities, root_num)
    else:
        post_post_word = np.zeros(all_len + 1)
    root = get_word_features(root_num, words, pos, feats, entities, -1)

    return np.concatenate((pre_pre_word, pre_word, word, post_word, post_post_word, root), axis=None)
	
def process_sentence(sentence, corpus_present=False, corpus_info=''):
    if corpus_present:  # get info from corpus if provided
        words, pos, feats, entities = get_info_from_corpus(corpus_info)
    else:  # get additional sentence info if not provided
        words, entities = get_words(sentence)  # TODO
        pos, feats = get_pos(sentence)  # TODO
        
    sentence_groups = []
    for word_num in range(len(words)):
        if pos[word_num] == 'NOUN' or pos[word_num] == 'PROPN':
            group = [str(word_num+1)]  # include the noun itself
            in_word_num = word_num
            while(in_word_num > 0):
                in_word_num -= 1
                word_features = get_features(in_word_num, words, pos, feats, entities, word_num)
                if word_in_group(word_features):
                    group.append(str(in_word_num+1))
                else:
                   break
            in_word_num = word_num
            while(in_word_num < len(words)-1):
                in_word_num += 1
                word_features = get_features(in_word_num, words, pos, feats, entities, word_num)
                if word_in_group(word_features):
                    group.append(str(in_word_num+1))
                else:
                   break

            group.sort(key=lambda string: int(string))
            sentence_groups.append(group)
            
    return sentence_groups
	
def word_in_group(word_features):
    return clf.predict(word_features.reshape(1, -1), batch_size=1)
	
transform_corpus('UD_Russian-SynTagRus-master/ru_syntagrus-ud-test.conllu', 'temp.txt')
with open('temp.txt', 'r', encoding='utf-8') as temp_f, open('Answers/test_given.txt', 'wb') as res_file:
    info = ''
    
    for line in temp_f:
        if len(line) <= 1:
            if len(info) > 0:
                res = process_sentence('', True, info)
                pickle.dump(res, res_file)
            info = ''
            continue

        parsed_line = line.split(maxsplit=3)
        if parsed_line[0] == '#':
            continue
        else:
            info = line[:-1]

    if len(info)>0:
        res = process_sentence('', True, info)
        pickle.dump(res, res_file)
		
def find_similar(group, answers):
    for answer in answers:
        if answer[0] == group[0] and len(answer) == len(group):
            res = True
            for i in range(len(answer)):
                if answer[i] != group[i]:
                    res = False
                    break
            if res:
                return True

    return False
	
def count_metrics(groups, answers):
    groups_count = len(groups)
    answers_count = len(answers)
    similar_count = 0

    for group in groups:
        if find_similar(group, answers):
            similar_count += 1

    precision = similar_count / groups_count if groups_count > 0 else 0
    recall = similar_count / answers_count if answers_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1
	
def metrics(given_answers_file, answers_file):
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
    count = 0

    with open(given_answers_file, 'rb') as f, open(answers_file, 'rb') as af:
        while True:  # read given_answers_file and answers_file line by line
            try:
                line = pickle.load(f)
            except EOFError:    # end of given_answers_file
                try:
                    _ = pickle.load(af)
                except EOFError:
                    break
                print('Answers file too long; ' + str(count) + ' lines read\n')
                break
            try:
                answer = pickle.load(af)
            except EOFError:    # end of answers_file
                print('Answers file too short; ' + str(count) + ' lines read\n')
                break

            precision, recall, f1 = count_metrics(line, answer)
            overall_precision += precision
            overall_recall += recall
            overall_f1 += f1
            count += 1

    if count == 0:
        print('No lines in file!\n')
        return 0, 0, 0

    precision = overall_precision / count
    recall = overall_recall / count
    f1 = overall_f1 / count
    return precision, recall, f1
	
def make_group(roots, noun):
    group = [noun]  # members of a group found so far
    candidates = [noun] # members of a group for which we search for dependent words

    while len(candidates) > 0:
        new_candidates = []
        for candidate in candidates:
            for pair in roots:
                if pair[1] == candidate:
                    group.append(pair[0])
                    new_candidates.append(pair[0])
        candidates = new_candidates # search for dependent words for new members of a group

    group.sort(key=lambda string: int(string))
    return group
	
def find_noun_groups(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as f, open(output_file, "wb") as outf:
        groups = []
        nouns = []
        roots = []

        for line in f:
            if len(line) <= 1:  # empty line: save groups for a previous sentence or skip
                if len(roots) > 0:
                    for noun in nouns:
                        groups.append(make_group(roots, noun))
                    pickle.dump(groups, outf)
                groups = []
                nouns = []
                roots = []
                continue
            line = line.split()
            if line[0] == '#':  # comment line: do nothing
                continue
            roots.append((line[0], line[6]))    # append a pair of (word, its root) to a list
            if line[3] == 'NOUN' or line[3] == 'PROPN':
                nouns.append(line[0])   # append a noun to a list of nouns in a sentence

        if len(roots) > 0: # save groups for a last sentence (if not saved already) or skip
            for noun in nouns:
                groups.append(make_group(roots, noun))
            pickle.dump(groups, outf)

    return
	
input_file = './UD_Russian-SynTagRus-master/ru_syntagrus-ud-test.conllu'
output_file = './Answers/test.txt'
#find_noun_groups(input_file, output_file)

answers_file = './Answers/test.txt'
given_answers_file = './Answers/test_given.txt'
print(metrics(given_answers_file, answers_file))
