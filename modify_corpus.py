from find_noun_groups import find_noun_groups
import pickle
import os


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
    with open(input_file, 'r', encoding='utf-8') as in_f, open(group_file, 'rb') as pf, \
	    open(output_file, 'w', encoding='utf-8') as out_f:
        roots = {}
		
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
            text = parsed_line[0] + ' ' + parsed_line[1] + ' ' + parsed_line[2] + ' ' + parsed_line[3] + ' ' + parsed_line[5] + ' '
            if parsed_line[0] in roots.keys():
                text += roots[parsed_line[0]]
            else:
                text += '0'            
            text = text + ' ' + parsed_line[7] + ' ' + parsed_line[8]
            print(text, file=out_f)
			

# usage example			
# input_file = './UD_Russian-SynTagRus-master/ru_syntagrus-ud-train.conllu'
# output_file = './UD_Russian-SynTagRus-master/train_reformed.conllu'
# temp_file = './temp1.pkl'
# temp_file_2 = './temp2.pkl'
# find_noun_groups(input_file, temp_file, True)
# prepare_sentences(temp_file, temp_file_2)
# modify_corpus(input_file, temp_file_2, output_file)
# os.remove('./temp1.pkl')
# os.remove('./temp2.pkl')
