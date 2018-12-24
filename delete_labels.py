import sys
			
			
# delete deprel labels from the original "CoNLL-U" format file line by line
			
def delete_labels(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        text = ''
        for line in in_f:
            if len(line) <= 1:  # empty line
                print(file=out_f)
                continue
            parsed_line = line.split()
            if parsed_line[0] == '#':  # text or info
                print(line[:-1], file=out_f)
                continue
            parsed_line = line.split('\t')
            text = parsed_line[0] + '\t' + parsed_line[1] + '\t' + parsed_line[2] + '\t' + parsed_line[3] + '\t' \
                   + parsed_line[4] + '\t' + parsed_line[5] + '\t' + parsed_line[6] + '\t'
            if parsed_line[0].find('.') > -1:
                text += '_\t'
            else:
                text += 'dep\t'
            parent = parsed_line[8].split(':')[0]
            text = text + parsed_line[8].split(':')[0] + ':dep'
            text = text + '\t' + parsed_line[9]
            print(text[:-1], file=out_f)
			

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    delete_labels(input_file, output_file)
