"""
Contains all the functions for cleaning news articles.
"""

import contractions
import inflect
# ALl the cleaning functions and utility that will be of used for us in the notebook.
import re
import unicodedata
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

download('stopwords')


# De-noise function
# Remove html links
def remove_between_square_brackets(text):
	"""Remove anything between brackets"""
	return re.sub('\[[^]]*\]', '', text)


# Remove http links
def remove_links(text):
	"""Remove http links in the text"""
	return re.sub('(https\S+|http\S+)', '', text)


# Replace contraction at this point will save us quite a bit of time later on
def replace_contractions(text):
	"""Replace contractions in string of text"""
	return contractions.fix(text)


# De-noise the text
def denoise_text(text):
	text = remove_between_square_brackets(text)
	text = remove_links(text)
	# text = replace_contractions(text)
	return text


# Normalize functions
def remove_non_ascii(words):
	"""Remove non-ASCII characters from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		new_words.append(new_word)
	return new_words


def to_lowercase(words):
	"""Convert all characters to lowercase from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = word.lower()
		new_words.append(new_word)
	return new_words


def remove_punctuation(words):
	"""Remove punctuation from list of tokenized words"""
	new_words = []
	for word in words:
		new_word = re.sub(r'[^\w\s]', '', word)
		if new_word != '':
			new_words.append(new_word)
	return new_words


def replace_numbers(words):
	"""Replace all integer occurrences in list of tokenized words with textual representation"""
	p = inflect.engine()
	new_words = []
	for word in words:
		if word.isdigit():
			new_word = p.number_to_words(word)
			new_words.append(new_word)
		else:
			new_words.append(word)
	return new_words


def remove_stopwords(words):
	"""Remove stop words from list of tokenized words"""
	new_words = []
	for word in words:
		if word not in stopwords.words('english'):
			new_words.append(word)
	return new_words


def stem_words(words):
	"""Stem words in list of tokenized words"""
	stemmer = LancasterStemmer()
	stems = []
	for word in words:
		stem = stemmer.stem(word)
		stems.append(stem)
	return stems


def lemmatize_verbs(words):
	"""Lemmatize verbs in list of tokenized words"""
	lemmatizer = WordNetLemmatizer()
	lemmas = []
	for word in words:
		lemma = lemmatizer.lemmatize(word, pos='v')
		lemmas.append(lemma)
	return lemmas


def normalize_text(words):
	words = remove_non_ascii(words)
	words = to_lowercase(words)
	words = remove_punctuation(words)
	# words = replace_numbers(words)
	words = remove_stopwords(words)
	return words


# Stemming and Lemmatize
def stem_and_lemmatize(words):
	# stems = stem_words(words)
	lemmas = lemmatize_verbs(words)
	return lemmas
