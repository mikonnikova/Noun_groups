import numpy as np
import pickle
import sys

import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed, merge


# train chunker with given features and answers
# save model to a file

def train_chunker(features_file, answers_file, model_file):

    # get training features
    with open(features_file, 'rb') as fx:
        x_train = pickle.load(fx)
        num_examples, seq_length, input_length = x_train.shape

    with open(answers_file, 'rb') as fy:
        y_train = pickle.load(fy)
        if y_train.shape[0] != num_examples:
            print('Inconsistent number of examples!')
            exit()

    # neural network input
    sequence = Input(shape=(seq_length, input_length), dtype='float32')

    # forwards LSTM
    forwards = LSTM(output_dim=50, return_sequences=True)(sequence)
    # backwards LSTM
    backwards = LSTM(output_dim=50, return_sequences=True, go_backwards=True)(sequence)

    # concatenate the outputs of the 2 LSTMs
    merged = merge([forwards, backwards], mode='concat')
    after_dp = Dropout(0.2)(merged)

    # output layer (softmax of 3)
    output = TimeDistributed(Dense(3, activation='softmax'))(after_dp)

    model = Model(input=sequence, output=output)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'],
                  sample_weight_mode='temporal')
    model.fit(x_train, y_train, batch_size=128, nb_epoch=50)

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
    features_file_name = sys.argv[2]
    answers_file_name = sys.argv[3]
    model_file_name = sys.argv[4]
    if sys.argv[1] == 'train':
        train_chunker(features_file_name, answers_file_name, model_file_name)
    elif sys.argv[1] == 'test':
        test_chunker(features_file_name, answers_file_name, model_file_name)
