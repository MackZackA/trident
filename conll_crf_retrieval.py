import pylint
import matplotlib.pyplot as plt
import nltk
import sklearn
import scipy.stats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# Using CoNLL2002 corpus for training
training_set = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_set = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

# Feature extraction
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.lower()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    
    return features

# Helper
def sent2features(sent):
    return [word2features(sent, a) for a in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# Training
x_train = [sent2features(x) for x in training_set]
y_train = [sent2labels(y) for y in training_set]
x_test = [sent2features(a) for a in test_set]
y_test = [sent2labels(b) for b in test_set]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
print('print crf fit:')
crf.fit(x_train, y_train)

# Evaluation
labels = list(crf.classes_)
eval = labels.remove('O')

# F1 measure
y_pred = crf.predict(x_test)
f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

# All four metrics
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))