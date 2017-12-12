import random

def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        #text = text.lower()
        data.append((label, text))
    return data

def read_testset(fname):
   with open(fname, 'r') as f:
      lines = f.readlines()
      lines = [line[1:] for line in lines]
      return lines

def text_to_trigrams(text):
    return ["%s%s%s" % (c1,c2, c3) for c1,c2,c3 in zip(text,text[1:], text[2:], text[3:], text[4:], text[5:])]

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]

test_sentences = read_testset("test")
TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]
TEST   = [text_to_bigrams(t) for t in test_sentences]


from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
voc_size = 600
vocab = set([x for x,c in fc.most_common(voc_size)])


vocab.add("UNKNOWN")

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
I2L = {i:l for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}
F2I['UNKNOWN'] = voc_size

