'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import io
import re
import warnings
import tarantool
import lib.chars as chars

##
# Filter Cython warnings about numpy changes
##
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

##
# Get data from Tarantool
##
srv = tarantool.connect(os.environ['TNT_HOST'], os.environ['TNT_PORT'])
text_samples = srv.space('sentences')

examples = text_samples.select(('Unknown', 'Nietzsche', 0), index=1, iterator=tarantool.const.ITERATOR_GE)

# The last one indicates whether the letter is capital
alphabet_len = len(chars.net_chars) + 1
seq_len = 40
seq_step = 3
sequences = []
next_chars = []

# build or load the model: a single LSTM
model = None
if len(sys.argv) > 1 and sys.argv[1] != '':
    print('Using model from ' + sys.argv[1])
    model = load_model(sys.argv[1])
else:
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, alphabet_len)))
    model.add(Dense(alphabet_len))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.save(os.environ['HOME'] + '/.keras/models/nitzsche.h5')

print('Vectorize dataset...')
for i, example in enumerate(examples):
    sentence = example[6]

    if len(sentence) < seq_len:
        sentence = ''.join((seq_step * 3 + seq_len - len(sentence)) * ['*'] + list(sentence))

    for i in range(0, len(sentence) - seq_len, seq_step):
        sequences.append(sentence[i: i + seq_len])
        next_chars.append(sentence[i + seq_len])

x = np.zeros((len(sequences), seq_len, alphabet_len), dtype=np.bool)
y = np.zeros((len(sequences), alphabet_len), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for j, c in enumerate(sequence):
        lower = c.lower()
        if lower in chars.char_index:
            x[i, j, chars.char_index[lower]] = 1
            if chars.is_uppercase(c):
                x[i, j, alphabet_len-1] = 1
    lower = next_chars[i].lower()
    if lower in chars.char_index:
        y[i, chars.char_index[lower]] = 1
        if chars.is_uppercase(next_chars[i]):
            y[i, alphabet_len-1] = 1


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


batch_size = 4096
refined_text = ' '.join(map(lambda s: s[6], examples))
def on_epoch_end(epoch, logs):
    model.save(os.environ['HOME'] + '/.keras/models/nitzsche_'+str(batch_size)+str(epoch)+'.h5')
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(refined_text) - seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = refined_text[start_index: start_index + seq_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, seq_len, alphabet_len))
            for t, char in enumerate(sentence):
                lower = char.lower()
                if lower in chars.char_index:
                    x_pred[0, t, chars.char_index[lower]] = 1.
                    if chars.is_uppercase(char):
                        x_pred[0, t, alphabet_len-1] = 1

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[0:alphabet_len-2], diversity)
            next_char = chars.index_char[next_index]
            if preds[alphabet_len-1] >= 0.5:
                next_char = next_char.upper()

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
        sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=batch_size,
          epochs=5,
          callbacks=[print_callback])
