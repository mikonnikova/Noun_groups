import pickle

# postprocessing of chunks
# one and only one noun
# beginning with B tag

def postprocessing(input_file, answer_file):
    answer = []

    with open(input_file, 'r', encoding='utf-8') as input_file:
        noun_in_chunk = 0
        sentence = {}
        chunk = []
        line_num = 0

        for line in input_file:
            if len(line) <= 1:  # empty line
                if len(chunk) > 0:
                    sentence.append(chunk)
                answer.append(sentence)
                noun_in_chunk = 0
                line_num = 0
                sentence = {}
                chunk = []
                continue

            line_num += 1
            parsed_line = line.split('\t')
            pos = (parsed_line[1] == 'NOUN' or parsed_line[1] == 'PROPN')
            tag = parsed_line[3][:-1]

            if tag == 'B':
                if len(chunk) > 0:
                    if noun_in_chunk > 0:  # add valid chunks only
                        sentence[noun_in_chunk] = chunk
                chunk = [line_num]
                if pos:
                    noun_in_chunk = line_num
                else:
                    noun_in_chunk = 0
            elif tag == 'I':
                if len(chunk) == 0:  # chunk with no beginning
                    continue
                if noun_in_chunk > 0 and pos:  # second noun in chunk
                    sentence[noun_in_chunk] = chunk
                    chunk = [line_num]
                    noun_in_chunk = line_num
                    continue
                chunk.append(line_num)
                if pos:
                    noun_in_chunk = line_num
            else:
                if len(chunk) > 0:
                    if noun_in_chunk > 0:  # add valid chunks only
                        sentence[noun_in_chunk] = chunk
                        noun_in_chunk = 0
                    chunk = []

    if len(chunk) > 0:
        sentence.append(chunk)
    answer.append(sentence)
    
    with open(answer_file, 'wb') as f:
        pickle.dump(answer, f)       
