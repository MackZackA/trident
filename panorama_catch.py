import string
import os
import collections
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import pos_tag, word_tokenize
from collections import Iterable

# Import corpus
ner_tags = collections.Counter()

corpus_root = "/home/ziang/Downloads/gmb-2.2.0/data"

for rt, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(rt, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
                    
                    standard_form_tokens = []

                    for index, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        # Attain only I and B-tagged tokens
                        if ner != 'O':
                            ner = ner.split('-')[0]
                        
                        ner_tags[ner] += 1


print(ner_tags)
print("Words=", sum(ner_tags.values()))

def features(tokens, index, history):
    """
    'tokens' = a POS-tagged sentence, in terms of a list of tuples [(w1, t1), ...]
    'index' = the index of the token we want to extract features for
    'history' = the IOB tags that are previously predicted
    """

    # Train the stemmer with English data, later converted to other languages
    stemmer = SnowballStemmer('english')
    
    # Initialize the tokens and history
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # Shift teh index with 2 so as to accomodate the initial placeholders
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prev2word, prev2pos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    next2word, next2pos = tokens[index + 2]
    previob = history[index - 1]
    with_dash = '-' in word
    with_dot = '.' in word
    all_ascii = all([True for i in word if i in string.ascii_lowercase])
    all_capitalized = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevall_capitalized = prevword == prevword.capitalize()
    prev_capitalized = prevword[0] in string.ascii_uppercase

    next_all_caps = prevword == prevword.capitalize()
    next_capitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all_ascii': all_ascii,
        'next_word': nextword,
        'all_ascii': all_ascii,

        'next_word': nextword,
        'next_lemma': stemmer.stem(nextword),
        'next_pos': nextpos,

        'next_next_word': next2word,
        'next_next_pos': next2pos,

        'prev_word': prevword,
        'prev_lemma': stemmer.stem(prevword),
        'prev_pos': prevpos,

        'prev_prev_word': prev2word,
        'prev_prev_pos': prev2pos,

        'prev_iob': previob,
        'with_dash': with_dash,
        'with_dot':with_dot,

        'all_capitalized': all_capitalized,
        'capitalized': capitalized,

        'prev_all_capitalized': prevall_capitalized,
        'prev_capitalized': prev_capitalized,

        'next_all_caps': next_all_caps,
        'next_capitalized': next_capitalized,
    }

def to_conll_iob(annotated_sentence):
    """
    'annotated_sentence' = a list of triplets represented as [(word1, token1, iob1, ...)]
    Transform a psudo-IOB sequence, denoted as "O, PERSON, PERSON, O, O, LOCATION, O"
    into proper IOB notation, denoted as "O, B-PERSON, I-PERSON, O, O, B-LOCATION, O"
    """
    iob_tokens = []
    for index, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if index == 0: # if current token is the first in the sentence
                ner = "B-" + ner
            elif annotated_sentence[index - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        iob_tokens.append((tag, word, ner))
    return iob_tokens

def read_corpus(corpus_root):
    for rt, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(rt, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
                        
                        standard_form_tokens = []

                        for index, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
                            
                            if ner != 'O':
                                ner = ner.split('-')[0]
                            
                            if tag in ('LQU', 'RQU'):
                                tag = "''"
                            
                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        """
                        Convert the training set to be compatible with formats of NLTK: [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        since the NLTK classifier is expected to take a tuple with the first item being the item input and second the class
                        """
                        yield [((w, t), iob) for w, t, iob in conll_tokens]

reader = read_corpus(corpus_root)

print(reader.__next__())
print('----------')

print(reader.__next__())
print('----------')

print(reader.__next__())
print('----------')

class NEChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs
        )
    
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the normalized format of triplets [(w1, t1, iob1), ...]
        iob_triplets = [(word, token, chunk) for ((word, token), chunk) in chunks]

        # Transformthe list of triplets to NLTK tree format
        return conlltags2tree(iob_triplets)

# Building the datasets
reader = read_corpus(corpus_root)
data = list(reader)
training_set = data[:int(len(data) * 0.9)]
test_set = data[:int(len(data) * 0.9):]

print("Training set = ", len(training_set))
print("Test set = ", len(test_set))

chunker = NEChunker(training_set[:2000])

# Show them steel
print(chunker.parse(pos_tag(word_tokenize("I am Leyasu Tokugawa from i2x"))))


# Evaluation
score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_set[:500]])
print("Accuracy:", score.accuracy())
