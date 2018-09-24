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
	
	
def find_noun_groups_for_trainset(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        #count = 0
        sentence_length = 0
        groups = []
        nouns = []
        roots = []
        total = []

        for line in f:
            #if count > 1000:
            #   break
            if len(line) <= 1:  # empty line: save groups for a previous sentence or skip
                #count += 1
                if len(roots) > 0:
                    for noun in nouns:
                        groups.append(make_group(roots, noun))
                    for gr in groups:
                        local = np.zeros(sentence_length)
                        for i in range(sentence_length):
                            if str(i) in gr:
                                local[i-1] = 1
                        total = np.append(total, local)
                    sentence_length = 0
                groups = []
                nouns = []
                roots = []
                continue
            line = line.split()
            if line[0] == '#':  # comment line: do nothing
                continue
            roots.append((line[0], line[6]))    # append a pair of (word, its root) to a list
            sentence_length += 1
            if line[3] == 'NOUN' or line[3] == 'PROPN':
                nouns.append(line[0])   # append a noun to a list of nouns in a sentence

        if len(roots) > 0: # save groups for a last sentence (if not saved already) or skip
            for noun in nouns:
                groups.append(make_group(roots, noun))
            for gr in groups:
                local = np.zeros(sentence_length)
                for i in range(sentence_length):
                    if str(i) in gr:
                        local[i-1] = 1
                total = np.append(total, local)

    return np.array(total)
	
	
# names of files
corpus = 'UD_Russian-SynTagRus-master/ru_syntagrus-ud-train.conllu'
output_name = 'Y_train.pkl'

Y = find_noun_groups_for_trainset(corpus)
print(Y.shape)
with open(output_name, 'wb') as f:
    pickle.dump(Y, f)