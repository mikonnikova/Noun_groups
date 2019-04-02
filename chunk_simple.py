import pickle


# delete inner chunk info from file

def to_chunked(conllu_file, chunk_file, output_file):

    with open(conllu_file, 'r', encoding='utf-8') as f, open(chunk_file, 'r', encoding='utf-8') as cf,\
    open(output_file, 'w') as of:
        for line in f:
            if len(line) <= 1 or line[0] == '#':  # empty line or info line
                print(line[:-1], file=of)
                continue

            parsed_line = line.split('\t')
            if parsed_line[0].find('.') > -1:  # ellipsis
                continue

            chunk_line = cf.readline()  # read next line in chunk file
            if len(chunk_line) <= 1:
                chunk_line = cf.readline()

            tag = chunk_line[-2]
            if tag == 'B' or tag == 'I':  # skip chunk member if not noun
                if parsed_line[3] == 'NOUN' or parsed_line[3] == 'PROPN':
                    print(line[:-1], file=of)
                else:
                    continue
            else:
                print(line[:-1], file=of)
                
    return


# add inner chuk info to answer

def from_chunked(answer_file, chunk_file, output_file):
    with open(answer_file, 'rb') as af, open(chunk_file, 'rb') as cf, open(output_file, 'wb') as of:
        while True:  # read files line by line
            try:
                answer = pickle.load(af)
            except EOFError:
                try:
                    _ = pickle.load(cf)
                except EOFError:
                    break
                print('Chunk file too long')
                break
            try:
                chunk = pickle.load(cf)
            except EOFError:
                print('Chunk file too short')
                break
                
            new_answer = {}
            for k,v in answer.items():
                new_answer[k] = v
                for item in v:
                    if item in chunk:
                        new_answer[k].append(chunk[item])
                        
            pickle.dump(new_answer, of)
            
    return
