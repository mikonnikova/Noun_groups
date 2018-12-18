import pickle
import sys


# find a group for a given noun
# takes in:
#    position of a noun
#    list of trios (word, its root, POS tag) for all words in a sentence
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
                    if pair[2] != 'PUNCT':
                        group.append(pair[0])
                    new_candidates.append(pair[0])
        candidates = new_candidates # search for dependent words for new members of a group

    group.sort(key=lambda string: int(string))
    return group


# find a shallow group for a given noun
# takes in:
#    position of a noun
#    list of trios (word, its root, POS tag) for all words in a sentence
# returns:
#    a shallow noun group (without inner ones) as a sorted by numbers list of positions of its members

def make_short_group(roots, noun):
    group = []  # members of a group found so far
    candidates = [noun] # members of a group for which we search for dependent words

    while len(candidates) > 0:
        new_candidates = []
        for candidate in candidates:
            for pair in roots:
                if pair[1] == candidate:
                    group.append(pair[0])
                    if pair[0] != 'NOUN' and pair[0] != 'PROPN':  # don't include embedded groups
                        new_candidates.append(pair[0])
        candidates = new_candidates # search for dependent words for new members of a group

    group.sort(key=lambda string: int(string))
    return group



# find noun groups in all of the sentences of a given file in "CoNLL-U" format
# save those groups in a pickle file sentence by sentence
# sentence format: dict of <noun, its group>

def find_noun_groups(input_file, output_file, short_version=False):
    with open(input_file, "r", encoding='utf-8') as f, open(output_file, "wb") as outf:
        groups = {}
        nouns = []
        roots = []

        for line in f:
            if len(line) <= 1:  # empty line: save groups for a previous sentence or skip
                if len(roots) > 0:
                    for noun in nouns:
                        if short_version:  # shallow or full noun groups
                            groups[noun] = make_short_group(roots, noun)
                        else:
                            groups[noun] = make_group(roots, noun)
                    pickle.dump(groups, outf)
                groups = {}
                nouns = []
                roots = []
                continue
            line = line.split()
            if line[0] == '#':  # comment line: do nothing
                continue
            roots.append((line[0], line[6], line[3]))    # append a trio of (word, its root, POS tag) to a list
            if line[3] == 'NOUN' or line[3] == 'PROPN':
                nouns.append(line[0])   # append a noun to a list of nouns in a sentence

        if len(roots) > 0: # save groups for a last sentence (if not saved already) or skip
            for noun in nouns:
                if short_version:  # shallow or full noun groups
                    groups[noun] = make_short_group(roots, noun)
                else:
                    groups[noun] = make_group(roots, noun)
            pickle.dump(groups, outf)

    return


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    find_noun_groups(input_file, output_file)
