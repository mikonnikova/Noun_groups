import numpy as np
import pickle
import sys

import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed, merge, Embedding
from keras.preprocessing.sequence import pad_sequences


# train chunker with given features and answers
# save model to a file

def train_chunker(words_file, pos_file, feats_file, answers_file, model_file):

    # get training features
    with open(words_file, 'rb') as wf:
        words_train = pickle.load(wf)
        num_examples = len(words_train)
        word_length = len(words_train[0][0])

    with open(pos_file, 'rb') as pf:
        pos_train = pickle.load(pf)
        if len(pos_train) != num_examples:
            print('Inconsistent number of examples!')
            exit()
        pos_length = len(pos_train[0][0])

    #with open(feats_file, 'rb') as ff:
    #    feats_train = pickle.load(ff)
    #    if len(feats_train) != num_examples:
    #        print('Inconsistent number of examples!')
    #        exit()
    #    feats_number = len(feats_train[0][0])
    #    feats_length = np.zeros(feats_number)
    #    for i in range(feats_number):
    #        feats_length[i] = len(feats_train[0][0][i])

    with open(answers_file, 'rb') as af:
        answers = pickle.load(af)
        if len(answers) != num_examples:
            print('Inconsistent number of examples!')
            exit()

    # neural network input
    words_sequence = Input(shape=(None, word_length), dtype='float32')
    pos_sequence = Input(shape=(None, ), dtype='float32')

    # pos embedding
    pos_embedding = Embedding(18, 10, input_length=None, mask_zero=True)(pos_sequence)

    # internal dropout
    in_dp_words = words_sequence
    in_dp_pos = Dropout(0.5)(pos_embedding)

    # internal merge
    merged1 = merge([in_dp_words, in_dp_pos], mode='concat')

    # forwards LSTM
    forwards = LSTM(output_dim=50, return_sequences=True)(merged1)
    # backwards LSTM
    backwards = LSTM(output_dim=50, return_sequences=True, go_backwards=True)(merged1)

    # concatenate the outputs of the 2 LSTMs
    merged2 = merge([forwards, backwards], mode='concat')
    after_dp = Dropout(0.5)(merged2)

    # output layer (softmax of 3)
    output = TimeDistributed(Dense(3, activation='softmax'))(after_dp)

    model = Model(input=[words_sequence, pos_sequence], output=output)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode='temporal')


    batch_size = 128
    for i in range(num_examples // batch_size):
        words = pad_sequences(words_train[i*128:(i+1)*128], padding='post')
        pos = pad_sequences(pos_train[i*128:(i+1)*128], padding='post')
        # feats = pad_sequences(feats_train[i*128:(i+1)*128], padding='post')
        answer = pad_sequences(answers[i*128:(i+1)*128], padding='post')

        for i in range(30):
            print(words.shape)
            print(pos.shape)
            model.train_on_batch([words, pos], answer)

    model.save(model_file)


# load model from a file
# get accuracy on a test file

def test_chunker(features_file, answers_file, model_file):

    # load model
    model = load_model(model_file)

    # get features and correct answers
    with open(features_file, 'rb') as fx:
        x_test = pickle.load(fx)
    with open(answers_file, 'rb') as fy:
        y_test = pickle.load(fy)

    score, acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', acc)
    # Test accuracy: 0.9580782173576496


if __name__ == '__main__':
    #features_file_name = sys.argv[2]
    #answers_file_name = sys.argv[3]
    #model_file_name = sys.argv[4]
    #if sys.argv[1] == 'train':
    #    train_chunker(features_file_name, answers_file_name, model_file_name)
    #elif sys.argv[1] == 'test':
    #    test_chunker(features_file_name, answers_file_name, model_file_name)
    words_file_name = './Chunk/dev_word_feats.pkl'
    pos_file_name = './Chunk/dev_pos_feats.pkl'
    feats_file_name = './Chunk/dev_feats_feats.pkl'
    answers_file_name = './Chunk/dev_answers.pkl'
    model_name = 'm.h5'
    train_chunker(words_file_name, pos_file_name, feats_file_name, answers_file_name, model_name)
