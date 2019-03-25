import numpy as np
import pickle
import sys
from sklearn.utils import shuffle

import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed, Embedding, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy


# train chunker with given features and answers
# save model to a file

def train_chunker(words_file, pos_file, answers_file, model_file, embedding_matrix_file):

    # get training features
    with open(words_file, 'rb') as wf:
        words_train = pickle.load(wf)
        num_examples = len(words_train)

    with open(pos_file, 'rb') as pf:
        pos_train = pickle.load(pf)
        if len(pos_train) != num_examples:
            print('Inconsistent number of examples!')
            exit()

    with open(answers_file, 'rb') as af:
        answers = pickle.load(af)
        if len(answers) != num_examples:
            print('Inconsistent number of examples!')
            exit()

    with open(embedding_matrix_file, 'rb') as ef:
        embedding_matrix = pickle.load(ef)

    # neural network input
    words_sequence = Input(shape=(None, ), dtype='float32')
    pos_sequence = Input(shape=(None, ), dtype='float32')

    # pos embedding
    word_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                               trainable=False, mask_zero=True)(words_sequence)
    post_word_embedding = Dense(15, activation='tanh')(word_embedding)
    pos_embedding = Embedding(18, 15, mask_zero=True)(pos_sequence)

    # internal dropout
    in_dp_words = Dropout(0.3)(post_word_embedding)
    in_dp_pos = Dropout(0.3)(pos_embedding)

    # internal merge
    merged1 = concatenate([in_dp_words, in_dp_pos])

    # forwards LSTM
    forwards = LSTM(units=15, return_sequences=True)(merged1)
    # backwards LSTM
    backwards = LSTM(go_backwards=True, units=15, return_sequences=True)(merged1)

    # concatenate the outputs of the 2 LSTMs
    merged2 = concatenate([forwards, backwards])
    after_dp = Dropout(0.3)(merged2)

    # output layer
    timed = TimeDistributed(Dense(10, activation='relu'))(after_dp)
    output = CRF(3)(timed)

    model = Model(inputs=[words_sequence, pos_sequence], outputs=output)
    model.summary()

    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_accuracy])

    batch_size = 16
    prev_loss = 1
    rounds_without_improvement = 0

    for _ in range(30):
        new_loss = 0

        # padding and training on batch
        for i in range(num_examples // batch_size - 1):
            words = pad_sequences(words_train[i * batch_size:(i + 1) * batch_size], padding='post')
            pos = pad_sequences(pos_train[i * batch_size:(i + 1) * batch_size], padding='post')
            answer = pad_sequences(answers[i * batch_size:(i + 1) * batch_size], padding='post')

            new_loss += model.train_on_batch([words, pos], answer)[0]

        i = num_examples // batch_size
        print(new_loss)

        # last batch
        if num_examples % batch_size > 0:
            words = pad_sequences(words_train[i * batch_size:], padding='post')
            pos = pad_sequences(pos_train[i * batch_size:], padding='post')
            answer = pad_sequences(answers[i * batch_size:], padding='post')

            new_loss += model.train_on_batch([words, pos], answer)[0]
            new_loss /= (i+1)

        else:
            new_loss /= i

        print(new_loss)
        # loss improvement
        if abs(prev_loss - new_loss) <= 1e-3:
            rounds_without_improvement += 1
        else:
            rounds_without_improvement = 0
        prev_loss = new_loss

        # early stopping
        if rounds_without_improvement >= 2:
            break

        # shuffle data
        words_train, pos_train, answers = shuffle(words_train, pos_train, answers)

    model.save(model_file)


if __name__ == '__main__':
    words_file_name = sys.argv[1]
    pos_file_name = sys.argv[2]
    answers_file_name = sys.argv[3]
    model_file_name = sys.argv[4]
    embedding_matrix_file_name = sys.argv[5]

    train_chunker(words_file_name, pos_file_name, answers_file_name, model_file_name, embedding_matrix_file_name)
