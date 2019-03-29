import pickle

# prepare an answer for chunker from IOB text file

def get_chunker_answer(input_file, output_file):
    answer = []  # array of dicts

    with open(input_file, 'r', encoding='utf-8') as f:
        sentence = {}
        chunk = []
        line_number = 0
        noun = 0
    
        for line in f:
            if len(line) <= 1:  # empty line
                if len(chunk) > 0:
                    sentence[noun] = chunk  # finish sentence
                answer.append(sentence)
                sentence = {}
                chunk = []
                line_number = 0  # start new sentence
                continue

            line_number += 1
            parsed_line = line.split('\t')
            pos = (parsed_line[1] == 'NOUN' or parsed_line[1] == 'PROPN')

            if parsed_line[3][:-1] == 'B':   # beginning of a chunk
                if len(chunk) > 0:
                    sentence[noun] = chunk
                chunk = [line_number]
                if pos:
                    noun = line_number
            elif parsed_line[3][:-1] == 'I':   # middle of a chunk
                chunk.append(line_number)
                if pos:
                    noun = line_number
            else:   # between chunks
                if len(chunk) > 0:
                    sentence[noun] = chunk
                    chunk = []

        if len(chunk) > 0:
            sentence[noun] = chunk  # finish sentence
        answer.append(sentence)
                
    with open(output_file, 'wb') as of:
        pickle.dump(answers, of)
        
    return
