#!/usr/bin/python3
# import libraries
from __future__ import absolute_import
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from six.moves import range
import operator
import nltk
import string
import sys
import re
import six


# read the text
f = open(sys.argv[1])

# Tokenize transcript
f2 = open(sys.argv[2])
nOpen_text = f2.read().replace("\n", " ").lower()
tokenizer = RegexpTokenizer(r'\w+')
tokenized = tokenizer.tokenize(nOpen_text)


# get the customized inputs for minimum character length, maximum word length, and number of top n words to extract
minmum_character_length = int(sys.argv[3])
maximum_words_length = int(sys.argv[4])
top_n = int(sys.argv[5])

# Preprocessing: if there is additional newline characters, replace it with a space-kill the empty lines
# read the whole script as a long string
open_text = f.read().replace("\n", " ")


def isPunct(word):
  return len(word) == 1 and word in string.punctuation

def isNumeric(word):
  try:
    float(word) if '.' in word else int(word)
    return True

  except ValueError:
    return False

# the threshold values collected from the user inputs
def threshold(phrase, min_char_length, max_words_length):
  # set a threshold value for the minimum character length a phrase has to have
  if len(phrase) < min_char_length:
    return 0 # if the phrase is too short then it is unable to pass

  # split the phrase to see if its number of words surpasses the maximum limit
  wordlist = phrase
  if len(wordlist) > max_words_length:
    return 0

  # check if the phrase has at least one alpha character
  digits = 0
  alpha = 0
  for i in range(0, len(phrase)):
    if phrase[i].isdigit():
      digits += 1
    elif phrase[i].isalpha():
      alpha += 1

  # checking if there is at least one character
  if alpha == 0:
    return 0

  # a phrase must have more alpha characters than digits
  if digits > alpha:
    return 0
  return 1


def match(phrase, tokenized_transcript): # take the phrase parameter as a string, while take the tokenized transcript as a list of words
  ps = PorterStemmer()
  tokenized_phrase = nltk.word_tokenize(phrase)
  tokenized_phrase = [word.lower() for word in tokenized_phrase]
  tokenized_phrase = [ps.stem(word) for word in tokenized_phrase]

  tp_length = len(tokenized_phrase)
  edge_length = 3
  search_length = tp_length + edge_length
  maximum_match = 0
  for idx in range(0, len(tokenized_transcript)-search_length+1):
    search_list = tokenized_transcript[idx:idx + search_length]
    search_list = [ps.stem(word) for word in search_list]

    match_number = 0
    txt_dict = {}
    tran_dict = {}
    for w in tokenized_phrase:
      if w not in txt_dict:
        txt_dict[w] = 1
      else:
        txt_dict[w] += 1

    for w in search_list:
      if w not in tran_dict:
        tran_dict[w] = 1
      else:
        tran_dict[w] += 1

    for word in txt_dict:
      if word in tran_dict:
        if txt_dict[word] > tran_dict[word]:
          match_number += tran_dict[word]
        else:
          match_number += txt_dict[word]
    if match_number > maximum_match:
      maximum_match = match_number


#  print("txt_dict", txt_dict)
#  print("tran_dict", tran_dict)

  matching_percentage = maximum_match / float(tp_length)
#  print("Max match:", maximum_match)
#  print("Match Number: ", match_number)
#  print("tp_length: ", tp_length)
  return matching_percentage


# Implementation of RAKE algorithm
class RakeKeywordExtractor:
  def __init__(self):
    self.stopwords = set(nltk.corpus.stopwords.words())
    self.top_fraction = 1 # consider top third candidate keywords by score

  def _generate_candidate_keywords(self, sentences, min_char_length = 1, max_words_length = 5):
    phrase_list = []
    for sentence in sentences:
      words = ["|" if x in self.stopwords else x for x in nltk.word_tokenize(sentence.lower())]

      phrase = []
      for word in words:
        if word == "|" or isPunct(word):
          # hey there are modifications here
          if len(phrase) > 0 and threshold(phrase, min_char_length, max_words_length):
            phrase_list.append(phrase)
            phrase = []
        else:
          phrase.append(word)

    return phrase_list


  def _calculate_word_scores(self, phrase_list):
    word_freq = nltk.FreqDist()
    word_degree = nltk.FreqDist()
    for phrase in phrase_list:
      degree = len(list([x for x in phrase if not isNumeric(x)])) - 1
      for word in phrase:
        word_freq[word] += 1
        word_degree[word] += degree # other words
    for word in list(word_freq.keys()):
      word_degree[word] = word_degree[word] + word_freq[word] # itself
    # word score = deg(w) / freq(w)
    word_scores = {}

    for word in list(word_freq.keys()):
      word_scores[word] = word_degree[word] / word_freq[word]

    return word_scores


  def _calculate_phrase_scores(self, phrase_list, word_scores):
    phrase_scores = {}
    for phrase in phrase_list:
      phrase_score = 0
      for word in phrase:
        phrase_score += word_scores[word]
      phrase_scores[" ".join(phrase)] = phrase_score

    return phrase_scores


  def extract(self, text, incl_scores=False):
    sentences = nltk.sent_tokenize(text)
    # modify the minimum character length and maximum words length here. I will recommend minimum_character_length = 1, maximum_words_length = 3.
    phrase_list = self._generate_candidate_keywords(sentences, minmum_character_length, maximum_words_length)
    word_scores = self._calculate_word_scores(phrase_list)
    phrase_scores = self._calculate_phrase_scores(
      phrase_list, word_scores)

    sorted_phrase_scores = sorted(iter(phrase_scores.items()),
      key = operator.itemgetter(1), reverse=True)

    n_phrases = len(sorted_phrase_scores)

    if incl_scores:
      return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]

    else:
      return [x[0] for x in sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]]


def test():
  rake = RakeKeywordExtractor()
  # extract the output as a list of tuples: the tuples are all pairs of phrases and their values
  keywords = rake.extract(open_text, incl_scores=True)
  output_list = keywords[:top_n]
  # printinput filename
  print("Input File: ", sys.argv[1])
  # print the top n most important key-word phrases in descending order
  sent_to_printout = 'The top ' + str(top_n) + ' most important phrases are: '
  print(sent_to_printout)
  print(output_list)
  print("The percentage of matching for each phrase is shown as followed:")
  for tup in output_list:
    match_percent = match(tup[0], tokenized)
    print(tup[0], match_percent)

  '''
  # FOR DEBUGGING, print out the whole list of extracted phrases, sorted by weighted values in descending sequence
  print("Print out the whole list of extracted phrases: ")
  print(processed_keywords) 
  print("End of Debugging")

  '''

if __name__ == "__main__":

  test()
