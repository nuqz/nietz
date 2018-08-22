import re
import chars

end_sentence = ''.join(chars.end_sentence_chars)
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
