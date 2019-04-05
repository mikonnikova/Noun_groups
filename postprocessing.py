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
        chunk_start = False

        for line in input_file:
            if len(line) <= 1:  # empty line
                if len(chunk) > 0:
                    if noun_in_chunk > 0:  # add valid chunks only
                        sentence[noun_in_chunk] = chunk
                answer.append(sentence)
                noun_in_chunk = 0
                line_num = 0
                sentence = {}
                chunk = []
                chunk_start = False
                continue

            line_num += 1
            parsed_line = line.split('\t')
            pos = (parsed_line[1] == 'NOUN' or parsed_line[1] == 'PROPN')
            punct = (parsed_line[1] == 'PUNCT')
            tag = parsed_line[3][:-1]

            if tag == 'B':
                if len(chunk) > 0:  # save previous chunk
                    if noun_in_chunk > 0:  # add valid chunks only
                        sentence[noun_in_chunk] = chunk
                if punct:
                    chunk = []
                    noun_in_chunk = 0
                else:
                    chunk = [line_num]
                    if pos:
                        noun_in_chunk = line_num
                    else:
                        noun_in_chunk = 0                        
                chunk_start = True
            elif tag == 'I':
                if not chunk_start:  # chunk with no beginning
                    chunk_start = True
                if punct:
                    continue
                if noun_in_chunk != 0 and pos:  # second noun in chunk
                    chunk = []
                    noun_in_chunk = -1
                    continue
                chunk.append(line_num)
                if pos:
                    noun_in_chunk = line_num
            else:
                if len(chunk) > 0:  # save previous chunk
                    if noun_in_chunk > 0:  # add valid chunks only
                        sentence[noun_in_chunk] = chunk
                noun_in_chunk = 0
                chunk = []
                chunk_start = False

    if len(chunk) > 0:
        if noun_in_chunk > 0:  # add valid chunks only
            sentence[noun_in_chunk] = chunk
    answer.append(sentence)
    
    with open(answer_file, 'wb') as f:
        print(answer)
        for e in answer:
            pickle.dump(e, f)    
        
    return


# postprocessing modified for model prediction

def postprocessing_modif(prediction, feats, answer_file):

    a_file = open(answer_file, 'ab')
    
    noun_in_chunk = 0
    sentence = {}
    chunk = []
    chunk_start = False

    for word_num in range(len(feats)):
        tag_code = prediction[word_num][0]
        if tag_code[0] == 1:
            tag = 'B'
        elif tag_code[1] == 1:
            tag = 'I'
        else:
            tag = 'O'

        noun = (feats[word_num] == 4 or feats[word_num] == 5)  # is noun
        punct = (feats[word_num] == 15)  # is punctuation

        if tag == 'B':
            if len(chunk) > 0:  # save previous chunk
                if noun_in_chunk > 0:  # add valid chunks only
                    sentence[noun_in_chunk] = chunk
            if punct:  # skip punctuation
                chunk = []
                noun_in_chunk = 0
            else:
                chunk = [word_num + 1]
                if noun:  # mark noun
                    noun_in_chunk = word_num + 1
                else:
                    noun_in_chunk = 0                        
            chunk_start = True
        elif tag == 'I':
            if not chunk_start:  # chunk with no beginning
                chunk_start = True
            if punct:  # skip punctuation
                continue
            if noun_in_chunk != 0 and noun:  # second noun in chunk
                chunk = []
                noun_in_chunk = -1
                continue
            chunk.append(word_num + 1)
            if noun:
                noun_in_chunk = word_num + 1
        else:
            if len(chunk) > 0:  # save previous chunk
                if noun_in_chunk > 0:  # add valid chunks only
                    sentence[noun_in_chunk] = chunk
            noun_in_chunk = 0
            chunk = []
            chunk_start = False

    if len(chunk) > 0:
        if noun_in_chunk > 0:  # add valid chunks only
            sentence[noun_in_chunk] = chunk
       
    pickle.dump(sentence, a_file)  
        
    return
