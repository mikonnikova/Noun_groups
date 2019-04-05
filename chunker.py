import numpy as np
import pickle
import sys
import os
from sklearn.utils import shuffle

import h5py
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed, Embedding, concatenate
from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from postprocessing import postprocessing_modif
from metrics import metrics


# train chunker with given features and answers
# save model to a file

def train_chunker(words_file, pos_file, answers_file, dev_words_file, dev_pos_file, dev_answers_file, model_file,
                  embedding_matrix_file, dense1, pos_emb, drop1, lstm, drop2, dense2, act1, act2, opt, batch):

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

    # get dev features
    with open(dev_words_file, 'rb') as wf:
        words_dev = pickle.load(wf)
    with open(dev_pos_file, 'rb') as pf:
        pos_dev = pickle.load(pf)

    with open(embedding_matrix_file, 'rb') as ef:
        embedding_matrix = pickle.load(ef)

    temp_file_name = model_file[:-2] + 'pkl'

    # neural network input
    words_sequence = Input(shape=(None, ), dtype='float32')
    pos_sequence = Input(shape=(None, ), dtype='float32')

    # pos embedding
    word_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                               trainable=False, mask_zero=True)(words_sequence)
    post_word_embedding = Dense(dense1, activation=act1)(word_embedding)
    pos_embedding = Embedding(18, pos_emb, mask_zero=True)(pos_sequence)

    # internal merge
    merged1 = concatenate([post_word_embedding, pos_embedding])
    in_dp = Dropout(drop1)(merged1)

    # forwards LSTM
    forwards = LSTM(units=lstm, return_sequences=True)(in_dp)
    # backwards LSTM
    backwards = LSTM(go_backwards=True, units=lstm, return_sequences=True)(in_dp)

    # concatenate the outputs of the 2 LSTMs
    merged2 = concatenate([forwards, backwards])
    after_dp = Dropout(drop2)(merged2)

    # output layer
    timed = TimeDistributed(Dense(dense2, activation=act2))(after_dp)
    output = CRF(3)(timed)

    model = Model(inputs=[words_sequence, pos_sequence], outputs=output)
    model.summary()

    model.compile(loss=crf_loss, optimizer=opt)

    batch_size = batch
    prev_quality = 0
    rounds_without_improvement = 0
    
    for e in range(50):
        loss = 0

        with open(model_file[:-2]+'txt', 'a') as info_file:
            print('Epoch ' + str(e), file=info_file)
            print(file=info_file)

        # padding and training on batch
        for i in range(num_examples // batch_size - 1):
            words = pad_sequences(words_train[i * batch_size:(i + 1) * batch_size], padding='post')
            pos = pad_sequences(pos_train[i * batch_size:(i + 1) * batch_size], padding='post')
            answer = pad_sequences(answers[i * batch_size:(i + 1) * batch_size], padding='post')

            loss += model.train_on_batch([words, pos], answer)

        i = num_examples // batch_size

        # last batch
        if num_examples % batch_size > 0:
            words = pad_sequences(words_train[i * batch_size:], padding='post')
            pos = pad_sequences(pos_train[i * batch_size:], padding='post')
            answer = pad_sequences(answers[i * batch_size:], padding='post')

            loss += model.train_on_batch([words, pos], answer)
            loss /= (i + 1)
        else:
            loss /= i

        with open(model_file[:-2]+'txt', 'a') as info_file:
            print('Loss: ' + str(loss), file=info_file)
            print(file=info_file)

        # get quality - metrics local
        for i in range(len(words_dev)):
            prediction = model.predict([words_dev[i], pos_dev[i]])
            postprocessing_modif(prediction, pos_dev[i], temp_file_name)
        quality, _ = metrics(temp_file_name, dev_answers_file, model_file[:-2]+'txt')  # get quality
        if os.path.exists(temp_file_name):  # delete temporary file
            os.remove(temp_file_name)

        # quality improvement
        if abs(quality - prev_quality) < 0.01:
            rounds_without_improvement += 1
        if quality >= prev_quality:
            if os.path.exists(model_file):  # delete previous model
                os.remove(model_file)
            model.save(model_file)  # save new model
        elif (prev_quality - quality) >= 0.1:
            break  # model degrading

        # early stopping
        if rounds_without_improvement >= 5:
            break

        # shuffle data
        words_train, pos_train, answers = shuffle(words_train, pos_train, answers, random_state=19)
        with open(model_file[:-2]+'txt', 'a') as info_file:
            print(file=info_file)

    # check on train set
    with open(model_file[:-2] + 'txt', 'a') as info_file:
        print("Train set", file=info_file)
    for i in range(len(words_train)):
        prediction = model.predict([words_train[i], pos_train[i]])
        postprocessing_modif(prediction, pos_dev[i], temp_file_name)
    quality, _ = metrics(temp_file_name, './train_chunk_answers.pkl', model_file[:-2] + 'txt')  # get quality
    # if os.path.exists(temp_file_name):  # delete temporary file
    #    os.remove(temp_file_name)

    return


if __name__ == '__main__':
    words_file_name = './train_word_feats.pkl'
    pos_file_name = './train_pos_feats.pkl'
    answers_file_name = './train_answers.pkl'
    dev_words_file_name = './dev_word_feats.pkl'
    dev_pos_file_name = './dev_pos_feats.pkl'
    dev_answers_file_name = './dev_chunks.pkl'  # prepared with chunker_correct_answer.py
    model_file_name = sys.argv[1]
    embedding_matrix_file_name = './embedding_matrix.pkl'

    dense1 = int(sys.argv[2])
    pos_emb = int(sys.argv[3])
    drop1 = float(sys.argv[4])
    lstm = int(sys.argv[5])
    drop2 = float(sys.argv[6])
    dense2 = int(sys.argv[7])
    act1 = sys.argv[8]
    act2 = sys.argv[9]
    opt = sys.argv[10]
    batch = int(sys.argv[11])

    train_chunker(words_file_name, pos_file_name, answers_file_name, dev_words_file_name, dev_pos_file_name,
                  dev_answers_file_name, model_file_name, embedding_matrix_file_name,
                  dense1, pos_emb, drop1, lstm, drop2, dense2, act1, act2, opt, batch)
