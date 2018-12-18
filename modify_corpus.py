from find_noun_groups import find_noun_groups
import pickle
import os
import sys


# read the pickle file with shallow noun groups sentence by sentence
# create a dict of pairs <word, its root> for all words in those groups
# write such dict for each sentence to a pickle file

def prepare_sentences(temp_file, temp_file_2):    
    with open(temp_file, 'rb') as f, open(temp_file_2, 'wb') as outf:
        while True:  # read file line by line
            try:
                line = pickle.load(f)
            except EOFError:    # end of given_answers_file
                break
            sentence_model = {}
            for k, v in line.items():
                for word in v:
                    sentence_model[word] = k
            pickle.dump(sentence_model, outf) 
			
			
# read the corpus file in "CoNLL-U" format line by line
# replace root information in original file with noun dependency information from pre-made sentence dict
			
def modify_corpus(input_file, group_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as in_f, open(group_file, 'rb') as pf, open(output_file, 'w', encoding='utf-8') as out_f:
        text = ''
        for line in in_f:
            if len(line) <= 1:  # empty line
                print(file=out_f)
                continue
            parsed_line = line.split()
            if parsed_line[0] == '#':  # text or info
                print(line[:-1], file=out_f)
                if parsed_line[1] == 'text':
                    roots = pickle.load(pf)
                continue
            parsed_line = line.split('\t')
            text = parsed_line[0] + '\t' + parsed_line[1] + '\t' + parsed_line[2] + '\t' + parsed_line[3] + '\t' \
                   + parsed_line[4] + '\t' + parsed_line[5] + '\t'  # first 6 columns unchanged           
            if parsed_line[0].find('.') > -1:
                text = text + '_\t_\t0:dep'  # dummy roots
            elif parsed_line[0] in roots.keys():
                text = text + roots[parsed_line[0]] + '\tdep\t' + roots[parsed_line[0]] + ':dep'
            else:
                text = text + '0\tdep\t0:dep'                 
            text = text + '\t' + parsed_line[9]
            print(text[:-1], file=out_f)
			

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    temp_file = './temp1.pkl'
    temp_file_2 = './temp2.pkl'
    find_noun_groups(input_file, temp_file, True)
    prepare_sentences(temp_file, temp_file_2)
    modify_corpus(input_file, temp_file_2, output_file)
    os.remove(temp_file)
    os.remove(temp_file_2)
