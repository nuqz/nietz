def char_range(a, z):
	for c in range(ord(a), ord(z)+1):
		yield(chr(c))

def char_list(a, z):
	return list(char_range(a, z))

space_chars = ['\n', ' ']
punctuation_chars = ['-', ',', ':', ';'] # TODO: add ', "
end_sentence_chars = ['.', '!', '?']
digit_chars = char_list('0', '9')
lowercase_letters = char_list('a', 'z')
uppercase_letters = char_list('A', 'Z')
auxiliary_chars = space_chars + punctuation_chars + end_sentence_chars
symbol_chars = lowercase_letters + uppercase_letters

net_chars = lowercase_letters + digit_chars + auxiliary_chars
char_index = dict((c, i) for i, c in enumerate(net_chars))
index_char = dict((i, c) for i, c in enumerate(net_chars))

def is_uppercase(c):
	return any(c in uc for uc in uppercase_letters)

allowed_chars = sorted(symbol_chars + digit_chars + auxiliary_chars)
def is_allowed(c):
	return any(c in ac for ac in allowed_chars)

def filter_chars(text):
	refined_text = ''	
	for c in text:
		if is_allowed(c):
			refined_text += c 
	return refined_text

