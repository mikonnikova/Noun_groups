import ufal.udpipe


# class for a simplified usage of udpipe.Model
# taken from https://github.com/ufal/udpipe/blob/master/bindings/python/examples/udpipe_model.py

class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output


# read text from a CoNLLU-format file and parse it with a given UDPipe model
# save results to a new file

def parse_text(model_file, input_file, output_file):
    model = Model(model_file)

    with open(input_file, 'r') as f, open(output_file, 'w') as of:
        for line in f:
            if len(line) <= 1:  # empty line, skip
                continue
            line = line.split(maxsplit=3)
            if line[0] == '#' and line[1] == 'text':    # text line, parse
                li = line[3]
                text = (li[:-1] + ' ')
                sentences = model.tokenize(text)
                for s in sentences:
                    model.tag(s)
                    model.parse(s)
                conllu = model.write(sentences, "conllu")
                print(conllu, file=of)


# postprocess CoNLLU file to glue together parts of a single sentence divided

def postprocess(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as of:
        offset = 0
        new_offset = 0
        parted = False

        for line in f:
            if len(line) <= 1:  # empty line
                continue
            temp_line = line.split()
            if temp_line[0] == '#':
                if temp_line[1] == 'sent_id' and temp_line[3] == '1':   # first part of a sentence
                    print(file=of)
                    print(line[:-1], file=of)
                    offset = 0
                    new_offset = 0
                    parted = False
                elif temp_line[1] == 'sent_id' and temp_line[3] != '1':  # part of a sentence divided
                    parted = True
                    if temp_line[3] != '2':
                        offset = new_offset
                continue
            if not parted:
                print(line[:-1], file=of)
                offset = int(line.split(maxsplit=1)[0])
            else:
                new_line = line.split(maxsplit=7)
                li = new_line[7]				
                print(str(int(new_line[0]) + offset) + ' ' + new_line[1] + ' ' + new_line[2] + ' ' + new_line[3] \
                    + ' ' + new_line[4] + ' ' + new_line[5] + ' ' + str(int(new_line[6]) + offset) + ' ' + \
					li[:-1], file=of)   # add offset to a number of a word and root
                new_offset = offset + int(new_line[0])
				


# usage example
# model_file = './udpipe-ud-2.0-170801/russian-syntagrus-ud-2.0-170801.udpipe'
# input_file = './ru_syntagrus-ud-test.conllu'
# temp_file = './temp.txt'
# output_file = './given_answer.txt'
# parse_text(model_file, input_file, temp_file)
# postprocess(temp_file, output_file)
