import pickle
import os


# delete inner chunk info from file
# uses conllu corpus and text chunks file
# returns modified corpus file and dict of new word numbers

def to_chunked(conllu_file, chunk_file, output_file, dict_file):
    with open(conllu_file, 'r', encoding='utf-8') as f, open(chunk_file, 'r') as cf, open(dict_file, 'wb') as df:
        num = 1
        sent_dict = {}

        # create dict of new words numbers
        for line in f:
            if len(line) <= 1 or line[0] == '#':  # empty line or info line
                if len(sent_dict) > 0:  # save sentence encoding
                    pickle.dump(sent_dict, df)
                sent_dict = {}
                num = 1
                continue

            parsed_line = line.split('\t', maxsplit=4)
            if parsed_line[0].find('.') > -1:  # ellipsis
                continue

            chunk_line = cf.readline()  # read next line in chunk file
            if len(chunk_line) <= 1:
                chunk_line = cf.readline()

            # get new numbers for words not inside chunks
            tag = chunk_line[-2]
            if tag == 'B' or tag == 'I':  # skip chunk member if not noun
                if parsed_line[3] == 'NOUN' or parsed_line[3] == 'PROPN':
                    sent_dict[str(num)] = parsed_line[0]
                    num += 1
                else:
                    continue
            else:
                sent_dict[str(num)] = parsed_line[0]
                num += 1

        if len(sent_dict) > 0:  # save sentence encoding
            pickle.dump(sent_dict, df)

    # get chunks to reform dependencies
    find_chunks(conllu_file, 'temp.pkl')
    prepare_chunked_sentences('temp.pkl', 'temp2.pkl')

    # reform file
    with open(conllu_file, 'r', encoding='utf-8') as f, open(dict_file, 'rb') as df, \
            open(output_file, 'w', encoding='utf-8') as of, open('temp2.pkl', 'rb') as cf:
        inverse_sent_dict = {}  # dict of word_number:new_word_number
        chunk = {}  # chunks for a given sentence

        for line in f:
            if len(line) <= 1 or line[0] == '#':  # empty line or info line
                if line[0] == '#' and line[2] == 't':  # get sentence encoding
                    sent_dict = pickle.load(df)
                    inverse_sent_dict = {}
                    for k, v in sent_dict.items():
                        inverse_sent_dict[v] = k
                    chunk = pickle.load(cf)
                print(line[:-1], file=of)
                continue

            parsed_line = line.split('\t')
            if parsed_line[0].find('.') > -1:  # ellipsis
                continue

            # get new text line with reformed numbers and dependencies
            if parsed_line[0] in inverse_sent_dict:
                text = inverse_sent_dict[parsed_line[0]] + '\t' + parsed_line[1] + '\t' + parsed_line[2] + '\t'
                text += (parsed_line[3] + '\t' + parsed_line[4] + '\t' + parsed_line[5] + '\t')
                root = chunk[parsed_line[6]] if parsed_line[6] in chunk else parsed_line[6]
                root = inverse_sent_dict[root] if root != '0' else '0'  # new root number
                text += (root + '\t' + parsed_line[7] + '\t')
                text += (root + ':' + parsed_line[7] + '\t')
                text += parsed_line[9][:-1]
                print(text, file=of)

    # remove temporary files
    os.remove('temp.pkl')
    os.remove('temp2.pkl')

    return


# add inner chunk info to answer
# uses given answer file, file of chunks and dict of new words numbers

def from_chunked(answer_file, chunk_file, output_file, dict_file):
    with open(answer_file, 'rb') as af, open(chunk_file, 'rb') as cf, open(output_file, 'wb') as of, \
            open(dict_file, 'rb') as df:
        while True:  # read files line by line
            try:
                answer = pickle.load(af)
                ch = pickle.load(cf)
                chunk = {}
                for k, v in ch.items():
                    chunk[str(k)] = [str(w) for w in v]
                sent_dict = pickle.load(df)
            except EOFError:
                break

            new_answer = {}
            for k, v in answer.items():  # for word in given answer
                new_group = []
                for item in v:
                    if str(sent_dict[item]) in chunk:  # add all chunk members to answer
                        new_group.extend(chunk[str(sent_dict[item])])
                    else:  # add original number of word to answer
                        new_group.extend(sent_dict[item])
                new_group.sort(key=lambda string: int(string))
                new_answer[sent_dict[k]] = new_group

            pickle.dump(new_answer, of)

    return
