# Imports
import nltk
import sys
import pylint
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import RegexpTokenizer
import re


# Classpath
'''
export STANFORDTOOLSDIR=$HOME
export CLASSPATH=$STANFORDTOOLSDIR/trident/stanford-ner-2015-04-20/stanford-ner.jar
export STANFORD_MODELS=$STANFORDTOOLSDIR/trident/stanford-ner-2015-04-20/classifiers
'''

# Read the texts
path_text = open(sys.argv[1]).read()
person_list = []
organization_list = []

p_phrase = []
o_phrase = []

f = path_text
f = f.replace('\n', ' ')
reg_tokenizer = RegexpTokenizer(r'\w+')
token_list = reg_tokenizer.tokenize(f)
tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

tag_sent = tagger.tag(token_list) # get a list of tuples
# use regex tokenizer to tokenize the text as a whole string, don't have to loop through each line
print("print tag sentence: ", tag_sent)
isNER = 0
for idx in range(0, len(tag_sent)):
    pair = tuple(tag_sent[idx])
    temp = isNER # store the flag value of previous iteration
    if 'PERSON' in pair[1]:
        isNER = 1
        if temp == isNER: # if this token has the same tag as its antecedent, append it in the p_phrase
            p_phrase.append(pair[0])
            print("print p_phrase: ", p_phrase)
            if idx == len(tag_sent) - 1: # the string has only one word
                    print("print p_phrase at the end of sentence", p_phrase)
                    ner = ' '.join(p_phrase)
                    print("print ner: ", ner)
                    person_list.append(ner)
                    p_phrase = []
        else:
            if idx == 0 and isNER == 1:
                p_phrase.append(pair[0])
                print("print p_phrase: first element", p_phrase)
                if idx == len(tag_sent) - 1: # the string has only one word
                    print("print p_phrase at the end of sentence", p_phrase)
                    ner = ' '.join(p_phrase)
                    print("print ner: ", ner)
                    person_list.append(ner)
                    p_phrase = []
            else:    
                p_phrase.append(pair[0])
                print("print p_phrase after collection", p_phrase)
    elif 'ORGANIZATION' in pair[1]:
        isNER = 2
        if temp == isNER:
            o_phrase.append(pair[0])
            print("print o_phrase: ", o_phrase)
            if idx == len(tag_sent) - 1: # the string reaches the sentence boundary
                    print("print o_phrase at the end of sentence", o_phrase)
                    ner = ' '.join(o_phrase)
                    print("print ner: ", ner)
                    organization_list.append(ner)
                    o_phrase = []
        else:
            if idx == 0 and isNER == 2:
                o_phrase.append(pair[0])
                print("print o_phrase: first element", o_phrase)
                if idx == len(tag_sent) - 1: # the string has only one word
                    print("print o_phrase at the end of sentence", o_phrase)
                    ner = ' '.join(o_phrase)
                    print("print ner: ", ner)
                    organization_list.append(ner)
                    o_phrase = []
            else:    
                o_phrase.append(pair[0])
                print("print o_phrase after collection", o_phrase)
    else:
        isNER = 0
        if len(p_phrase) != 0:
            print("print p_phrase before collecting", p_phrase)
            ner = ' '.join(p_phrase)
            print("print ner: ", ner)
            person_list.append(ner)
            p_phrase = []
        if len(o_phrase) != 0:
            print("print o_phrase: ", o_phrase)
            ner = ' '.join(o_phrase)
            print("print ner:", ner)
            organization_list.append(ner)
            o_phrase = []

        continue

print("Names of people: ", person_list)
print("Names of organizations: ", organization_list)