import os
import re
import sys
import numpy as np
import tarantool
import lib.chars as chars
import lib.text as txt

##
# Connect to tarantool database
##
srv = tarantool.connect(os.environ['TNT_HOST'], os.environ['TNT_PORT'])
text_samples = srv.space('sentences')

##
# Read example file
##
sample_file = open(sys.argv[1], 'r')
text = sample_file.read()
sample_file.close()

##
# Remove numbers in front of paragraph (150. Somthing goes wrong...)
# Remove annotations and their numbers ('[0]' and '[0] Describe something')
# Remove -- (e.g. picture--that)
# Unwrap by pattern like =No Within and Without in the World.=
##
text = re.sub('\n\d+\.\s?([^'+txt.end_sentence+']+['+txt.end_sentence+'])', '\\1', 
	re.sub('\[\d\](\s?[^'+txt.end_sentence+']+['+txt.end_sentence+'])?', '',
		re.sub('=([^.=]+).?=', '\\1', 
			re.sub('--', ' ', text))))

##
# Remove redundant characters
##
def fix_dots(text):
	return re.sub('\.\s+\.\s', '. ',
		re.sub('(\w)\.(\w)', '\\1. \\2',
			re.sub('(\w)\n\n', '\\1.\n', text)))

def remove_page_nums(text):
	return re.sub('\d{1,3}', '', text)

def fix_repeating_auxiliary(text):
	for pc in chars.auxiliary_chars:
		text = re.sub('([' + pc + ']){2,}', '\\1', text)
	return text

refined_text = re.sub('\n', ' ', fix_repeating_auxiliary(
	fix_dots(remove_page_nums(chars.filter_chars(text)))))

sentences = dict((i, s) for i, s in enumerate(txt.find_sentences(refined_text)))
statements = txt.find_statements(refined_text)
questions = txt.find_questions(refined_text)
exclamations = txt.find_exclamations(refined_text)

def is_in_list(sentence, lst):
	return any(sentence in s for s in lst)

for i, sentence in sentences.iteritems():
	sentence_type = 0
	if is_in_list(sentence, questions):
		sentence_type = 1
	elif is_in_list(sentence, exclamations):
		sentence_type = 2
	text_samples.insert((None, 'Unknown', 'Nietzsche', 1825,
		sentence_type, i, sentence.strip()))

