import pickle


# NP chunk includes a continuous string of words and punctuation marks in the sentence
# it omits any embedded noun groups
# to form a chunk a shallow noun group is initially formed, then all the gaps in it are processed

def make_chunk(roots, noun):
    group = []  # members of a shallow noun group found so far
    candidates = [noun]  # members of a group for which we search for dependent words

    while len(candidates) > 0:
        new_candidates = []
        for candidate in candidates:
            for info in roots:
                if info[1] == candidate:
                    if info[2] != 'NOUN' and info[2] != 'PROPN':  # do not include embedded noun groups
                        group.append(info[0])
                        new_candidates.append(info[0])
        candidates = new_candidates  # search for dependent words for new members of a group
    group.append(noun)  # include the main noun itself

    chunk = []
    word_number = int(noun)
    # include continuous string of words to the left and to the right of main noun
    while word_number > 0 and str(word_number) in group:
        chunk.append(str(word_number))
        word_number -= 1
    word_number = int(noun) + 1
    while str(word_number) in group:
        chunk.append(str(word_number))
        word_number += 1
    # sort the chunk in ascending order
    chunk.sort(key=lambda string: int(string))

    return chunk


# for each sentence in a CONLL-U format file return a dict of NP chunks
# in a format noun:(its chunk)

def find_chunks(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as f, open(output_file, "wb") as outf:
        chunks = {}
        nouns = []
        roots = []

        for line in f:
            if len(line) <= 1:  # empty line: save chunks for a previous sentence or skip
                if len(roots) > 0:
                    for noun in nouns:
                        chunks[noun] = make_chunk(roots, noun)
                    pickle.dump(chunks, outf)
                chunks = {}
                nouns = []
                roots = []
                continue
            if line.split()[0] == '#':  # comment line: do nothing
                continue
            line = line.split('\t')
            roots.append((line[0], line[6], line[3]))    # append a trio of (word, its root, POS tag) to a list
            if line[3] == 'NOUN' or line[3] == 'PROPN':
                nouns.append(line[0])   # append a noun to a list of nouns in a sentence

        if len(roots) > 0:  # save chunks for a last sentence (if not saved already) or skip
            for noun in nouns:
                chunks[noun] = make_chunk(roots, noun)
            pickle.dump(chunks, outf)

    return
