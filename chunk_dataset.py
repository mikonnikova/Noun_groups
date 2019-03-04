from find_chunks import find_chunks
import pickle
import os
import sys


# take a temporary file of dicts of chunks in the sentences of original file
# for each sentence prepare a dict of pairs word:its_chunk (main noun)

def prepare_chunked_sentences(temp_file, temp_file_2):
    with open(temp_file, 'rb') as f, open(temp_file_2, 'wb') as outf:
        while True:  # read file line by line
            try:
                line = pickle.load(f)
            except EOFError:    # end of file
                break
            sentence_model = {}
            for k, v in line.items():
                for word in v:
                    sentence_model[word] = k
            pickle.dump(sentence_model, outf)


# write a text file of info for words in a sentence
# word POS_tag FEATS_tag chunk_label(IOB2)
# sentences are separated by empty lines

def write_chunks(input_file, chunk_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as in_f, open(chunk_file, 'rb') as pf, \
            open(output_file, 'w', encoding='utf-8') as out_f:

        for line in in_f:
            if len(line) <= 1:  # empty line
                print(file=out_f)
                continue
            parsed_line = line.split()
            if parsed_line[0] == '#':  # text or info
                if parsed_line[1] == 'text':
                    roots = pickle.load(pf)
                    root = '0'
                continue
            parsed_line = line.split('\t')
            if parsed_line[0].find('.') > -1:
                continue  # ellipsis
            text = parsed_line[2] + '\t' + parsed_line[3] + '\t' + parsed_line[5]
            if parsed_line[0] in roots.keys():
                if roots[parsed_line[0]] != root:
                    text += '\tB'
                else:
                    text += '\tI'
                root = roots[parsed_line[0]]
            else:
                text += '\tO'
            print(text, file=out_f)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    temp_file = './temp1.pkl'
    temp_file_2 = './temp2.pkl'
    find_chunks(input_file, temp_file)
    prepare_chunked_sentences(temp_file, temp_file_2)
    write_chunks(input_file, temp_file_2, output_file)
    os.remove(temp_file)
    os.remove(temp_file_2)
