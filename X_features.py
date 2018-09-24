import gensim
import numpy as np
import pickle

m = 'ruscorpora_upos_skipgram_300_5_2018.vec'  # file with used word embedding model
model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)  # load word embedding model

# dictionary of part-of-speech tags
pos_dict = {'ADJ':1, 'ADV':2, 'INTJ':3, 'NOUN':4, 'PROPN':5, 'VERB':6, 'ADP':7, 'AUX':8, 'CCONJ':9, 'DET':10, 'NUM':11, \
            'PART':12, 'PRON':13, 'SCONJ':14, 'PUNCT':15, 'SYM':16, 'X':17}

# variants of additional part-of-speech features
feats_variants = {'Foreign':{'Yes':1},
                  'Gender':{'Fem':1, 'Masc':2, 'Neut':3},
                  'Animacy':{'Anim':1, 'Inan':2},
                  'Number':{'Plur':1, 'Sing':2},
                  'Case':{'Acc':1, 'Nom':2, 'Dat':3, 'Gen':4, 'Ins':5, 'Loc':6, 'Par':7, 'Voc':8},
                  'VerbForm':{'Conv':1, 'Fin':2, 'Inf':3, 'Part':4},
                  'Mood':{'Ind':1, 'Imp':2, 'Cnd':3},
                  'Tense':{'Fut':1, 'Past':2, 'Pres':3},
                  'Person':{'1':1, '2':2, '3':3},
                  'Aspect':{'Imp':1, 'Perf':2},
                  'Voice':{'Act':1, 'Mid':2, 'Pass':3}}

# numbers of variants of additional part-of-speech features
feats_lengths = {}
for f, d in feats_variants.items():
    feats_lengths[f] = len(d) + 1

word_emb_len = len(model.get_vector('студент_NOUN')) + 1  # length of word embedding vector feature
pos_len = len(pos_dict) + 1  # length of part-of-speech vector feature
feats_len = 0  # length of additional part-of-speech features vector feature
for f, l in feats_lengths.items():
    feats_len += int(l)
all_len = word_emb_len + pos_len + feats_len + 1  # total length of a features vector for a single word


# reads input CoNLL-U file
# ouputs text and selected word features to the output file

def transform_corpus(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as ff:
        text = ''
        for line in f:
            if len(line) <= 1:  # go to the next text feature
                if len(text) > 0:
                    print(text, file=ff)
                    text = ''
                print(file=ff)
                continue
            parsed_line = line.split()
            if parsed_line[0] == '#':
                if parsed_line[1] == 'text':  # get given sentence
                    print(line[:-1], file=ff) 
                continue
            text = text + parsed_line[0] + ' ' + parsed_line[1] + ' ' + parsed_line[2] + ' ' + parsed_line[3] + ' ' \
                    + parsed_line[5] + ' '  # add word information
					

# organize given corpus info into separate lists
def get_info_from_corpus(corpus_info):
    words = []
    pos = []
    feats = []
    entities = []
    
    info = corpus_info.split()
    for word_count in range(len(info)//5):
        words.append(info[word_count*5 + 2])  # lemmas
        pos.append(info[word_count*5 + 3])  # part-of-speech tags
        feats.append(info[word_count*5 + 4])  # additional part-of-speech features
        # get words starting with title letters (entities candidates)
        if word_count>0 and info[word_count*5 + 1].istitle():
            entities.append(1)
        else:
            entities.append(0)
    
    return words, pos, feats, entities
	
	
# transform given word inforamtion into a vector representation
def get_word_features(num, words, pos, feats, entities):
    word_pos = words[num] + '_' + pos[num]
    # get word embedding from a model
    word_vector = np.concatenate(([1], model.get_vector(word_pos)), axis=None) if word_pos in model else np.zeros(word_emb_len) 
    
    # add part-of-speech feature one-hot encoding
    pos_vector = np.zeros(pos_len)
    if pos[num] in pos_dict:
        pos_vector[pos_dict[pos[num]]] = 1
        pos_vector[0] = 1
    
    # add part-of-speech additional features one-hot encodings
    feats_list = [f.split('=') for f in feats[num].split('|')]
    feats_dict = {}
    for f in feats_list:
        if len(f) > 1:
            feats_dict[f[0]] = f[1]        
    feats_vector = np.zeros(0)
    for a in feats_lengths.keys():
        local = np.zeros(feats_lengths[a])
        if a in feats_dict:
            local[0] = 1
            local[feats_variants[a][feats_dict[a]]] = 1
        feats_vector = np.append(feats_vector, local)
    
    return np.concatenate(([1], word_vector, pos_vector, feats_vector, entities[num]), axis=None)
	
	
# get a global features vector for a word
# comprising info about word in a 2-word window and a root word
def get_features(num, words, pos, feats, entities, root_num):
    if num >= 2:
        pre_pre_word = get_word_features(num-2, words, pos, feats, entities)
    else:
        pre_pre_word = np.zeros(all_len+1)
    if num >= 1:
        pre_word = get_word_features(num-1, words, pos, feats, entities)
    else:
        pre_word = np.zeros(all_len+1)
    word = get_word_features(num, words, pos, feats, entities)
    if num <= len(words)-2:
        post_word = get_word_features(num+1, words, pos, feats, entities)
    else:
        post_word = np.zeros(all_len+1)
    if num <= len(words)-3:
        post_post_word = get_word_features(num+2, words, pos, feats, entities)
    else:
        post_post_word = np.zeros(all_len+1)
    root = get_word_features(root_num, words, pos, feats, entities)
    
    return np.concatenate((pre_pre_word, pre_word, word, post_word, post_post_word, root), axis=None)
	

# for a given sentence with additional corpus info get all feature vectors
def get_sentence_features(sentence, corpus_info):
    words, pos, feats, entities = get_info_from_corpus(corpus_info)
    all_features = []
        
    sentence_groups = []
    for word_num in range(len(words)):
        if pos[word_num] == 'NOUN' or pos[word_num] == 'PROPN':
            for in_word_num in range(len(words)):  # check all words in a sentence
                word_features = get_features(in_word_num, words, pos, feats, entities, word_num)
                all_features.append(word_features)
            
    return np.array(all_features)
	
	
# names of files
corpus = 'UD_Russian-SynTagRus-master/ru_syntagrus-ud-train.conllu'
output_name = 'X_train.pkl'

transform_corpus(corpus, 'temp.txt')
X = np.zeros((1, (all_len+1)*6), dtype=int)
with open('temp.txt', 'r', encoding='utf-8') as f:
    #count = 0
    text = ''
    info = ''
    for line in f:
        #if count > 1000:
        #    break
        if len(line) <= 1:
            #count += 1
            if len(text)>0:
                temp = get_sentence_features(text, info)
                if len(temp)>0:
                    X = np.concatenate((X, temp), axis=0)
            text = ''
            info = ''
            continue
        
        parsed_line = line.split(maxsplit=3)
        if parsed_line[0] == '#':
            text = parsed_line[3]
            text = text[:-1]
        else:
            info = line[:-1]

temp = get_sentence_features(text, info)
if len(temp)>0:
    X = np.concatenate((X, temp), axis=0)
X = X[1:, :]
with open(output_name, 'wb') as f:
    pickle.dump(X, f)
print(X.shape)