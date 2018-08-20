import re
import numpy as np

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

print(char_index)
print(index_char)
