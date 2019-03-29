import pickle
from keras.models import load_model
from get_chunker_features import process_sentence

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

# predict chunk tags from model to text file

def predict_tags(test_file, answer_file, model_file):

    model = load_model(model_file, custom_objects={'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy})

    with open(test_file, 'r', encoding='utf-8') as input_file, open(answer_file, 'w') as output_file:
        line_num = 0
        sent_num = 0

        for line in test_file:
            data1, data2 = process_sentence(line)
            answer = model.predict([data1, data2])
            
            if len(line) <= 1:
                print(file=output_file)
                sent_num += 1
                line_num = 0
                continue
            tag_code = answer[sent_num][line_num]
            if tag_code[0]:
                tag = 'B'
            elif tag_code[1]:
                tag = 'I'
            else:
                tag = 'O'
            print(line[:-2] + tag, file=output_file)
            line_num += 1
            
    return
