import pickle


# find a group for a given noun
# takes in:
#    position of a noun
#    list of pairs (word, its root) for all words in a sentence
# returns:
#    a noun group as a sorted by numbers list of positions of its members

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


# find noun groups in all of the sentences of a given file in "CoNLL-U" format
# save those groups in a pickle file

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


# usage example
# input_file = './UD_Russian-SynTagRus-master/ru_syntagrus-ud-test.conllu'
# output_file = './Answers/test.txt'
# find_noun_groups(input_file, output_file)
