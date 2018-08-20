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
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import re
import warnings

##
# Filter Cython warnings about numpy changes
##
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

##
# Read example file
##
sample_file = open("/home/nuqz/.keras/datasets/nietzsche.txt", 'r')
text = sample_file.read()
sample_file.close()

##
# TODO:
# Split by chapters (x)
    # Split by paragraphs (y)
        # Split by sentences (z)
            # Split by shingles2 (sx)
                # Split by words (x, y, z, sx, sy, sz, sd)
            # Split by shingles3 (sy)
                # Split by words (x, y, z, sx, sy, sz, sd)
            # Split by shingles4 (sz)
                # Split by words (x, y, z, sx, sy, sz, sd)
            # Split by shingles5 (sd)
                # Split by words (x, y, z, sx, sy, sz, sd)
#
##

def char_range(a, z):
    for c in range(ord(a), ord(z)+1):
        yield(chr(c))

def char_list(a, z):
    return list(char_range(a, z))

space_chars = ['\n', ' ']
punctuation_chars = ['-', ',', ':', ';'] # TODO: add ', "
end_sentence_chars = ['.', '!', '?']
auxiliary_chars = space_chars + punctuation_chars + end_sentence_chars
symbol_chars = char_list('a', 'z') + char_list('A', 'Z')
digit_chars = char_list('0', '9')

##
# Remove numbers in front of paragraph (150. Somthing goes wrong...)
# Remove annotations and their numbers ('[0]' and '[0] Describe something')
# Remove -- (e.g. picture--that)
# Unwrap by pattern like =No Within and Without in the World.=
##
end_sentence = ''.join(end_sentence_chars)
text = re.sub('\n\d+\.\s?([^'+end_sentence+']+['+end_sentence+'])', '\\1', 
    re.sub('\[\d\](\s?[^'+end_sentence+']+['+end_sentence+'])?', '',
        re.sub('=([^.=]+).?=', '\\1', 
            re.sub('--', ' ', text))))

##
# Remove redundant characters
##
allowed_chars = sorted(symbol_chars + digit_chars + auxiliary_chars)
char_index = dict((c, i) for i, c in enumerate(allowed_chars))
index_char = dict((i, c) for i, c in enumerate(allowed_chars))

def is_allowed(c):
    return any(c in ac for ac in allowed_chars)

def fix_dots(text):
    return re.sub('\.\s+\.\s', '. ',
        re.sub('(\w)\.(\w)', '\\1. \\2',
            re.sub('(\w)\n\n', '\\1.\n', text)))

def remove_page_nums(text):
    return re.sub('\d{1,3}', '', text)

def fix_repeating_auxiliary(text):
    for pc in auxiliary_chars:
        text = re.sub('([' + pc + ']){2,}', '\\1', text)
    return text

refined_text = ''   
for c in text:
    if is_allowed(c):
        refined_text += c 

refined_text = re.sub('\n', ' ', fix_repeating_auxiliary(
    fix_dots(remove_page_nums(refined_text))))

##
# Vectorize inputs
##
common_sentence_part = '[^' + end_sentence + ']+'

sentences_re = re.compile(common_sentence_part + \
    '[' + end_sentence + ']\s?')
def find_sentences(text):
    return sentences_re.findall(text)

statements_re = re.compile(common_sentence_part + '[.]\s?')
def find_statements(text):
    return statements_re.findall(text)

questions_re = re.compile(common_sentence_part + '[?]\s?')
def find_questions(text):
    return questions_re.findall(text)

exclamations_re = re.compile(common_sentence_part + '[!]\s?')
def find_exclamations(text):
    return exclamations_re.findall(text)
statements = find_statements(refined_text)
def is_statement(sentence):
    return any(sentence in s for s in statements)
# print(statements)

questions = find_questions(refined_text)
def is_question(sentence):
    return any(sentence in q for q in questions)
# print(questions)

exclamations = find_exclamations(refined_text)
def is_exclamation(sentence):
    return any(sentence in e for e in exclamations)
# print(exclamations)

seq_len = 40
seq_step = 3
sequences = []
next_chars = []

sentences = find_sentences(refined_text)
for i, sentence in enumerate(sentences):
    # Insert into tarantool
    sentence_type = 0
    if is_question(sentence):
        sentence_type = 1
    elif is_exclamation(sentence):
        sentence_type = 2

    sentence = sentence.strip()
    if (len(sentence) < seq_len) and (i != 0):
        sentence = sentences[i-1].strip() + ' ' + sentence
    for i in range(0, len(sentence) - seq_len, seq_step):
        if len(sentence) < seq_len:
            print(i, sentence[i: i+seq_len])
        sequences.append(sentence[i: i + seq_len])
        next_chars.append(sentence[i + seq_len])

x = np.zeros((len(sequences), seq_len, len(allowed_chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(allowed_chars)), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for j, c in enumerate(sequence):
        x[i, j, char_index[c]] = 1
    y[i, char_index[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(seq_len, len(allowed_chars))))
model.add(Dense(len(allowed_chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


batch_size = 64
def on_epoch_end(epoch, logs):
    model.save('/home/nuqz/.keras/models/nitzsche_'+str(epoch)+str(batch_size)+'.h5')
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
            x_pred = np.zeros((1, seq_len, len(allowed_chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_index[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_char[next_index]

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
