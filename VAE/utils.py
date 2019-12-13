import numpy as np
import pandas as pd
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict,Counter

import gensim
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk import pos_tag
wnl = WordNetLemmatizer() 
porter = PorterStemmer()

from nltk.corpus import stopwords
words = set(nltk.corpus.words.words())
stops = set(stopwords.words('english'))

from spacy.lang.en import English
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

def process_text(texts, bigram, with_bigram=True,filter_stopword_spacy = False, remove_non_eng = False,
                    with_lemmatize = False, with_stem = False):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal(nltk & spacy)
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    4. Stem
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    # stopword removal using NLTK's english stopwords dataset remove non-english words.
    texts = [[word for word in line if word not in stops and word.isalpha() ] for line in texts] 
    if remove_non_eng:
        texts = [[word for word in line if word.lower() in words] for line in texts]
    # Bigram collocation detection (frequently co-occuring tokens) using gensim's Phrases. can even try trigram collocation detection.
    if with_bigram:
        bigram = gensim.models.Phrases(texts)
        texts = [bigram[line] for line in texts]
    #lemmatization (using gensim's lemmatize) to only keep the nouns. Lemmatization is generally better than stemming in the case of topic modeling since the words after lemmatization still remain understable. However, generally stemming might be preferred if the data is being fed into a vectorizer and isn't intended to be viewed. 
    if with_lemmatize:
        lemmatization = ['a','n','v']
        texts = [[wnl.lemmatize(i,j[0].lower()) if j[0].lower() in lemmatization else wnl.lemmatize(i) for i,j in pos_tag(line)] for line in texts]
    if with_stem:
        texts = [[porter.stem(word) for word in line] for line in texts] 
    if filter_stopword_spacy:
        texts = [[word for word in line if not nlp.vocab[word].is_stop] for line in texts] 
    return texts

def build_vocab(all_tokens, max_vocab_size):
    '''
    Returns:
     id2token, token2id
    ''' 
    PAD_IDX = 0
    UNK_IDX = 1
    token_counter = Counter(all_tokens)
    # unzip the vocab and its corresponding count
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    # Give indices from 2 to the vocab
    token2id = dict(zip(vocab, range(2, 2+len(vocab))))
    # Add pad and unk to vocab
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def token2index_dataset(tokens_data, token2id, id2token):
    UNK_IDX = 1
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def create_minibatch(data,batch_size):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]

def cal_val_ppl(model,val):
    cost=[]
    for doc in val:
        doc = doc.astype('float32')
        n_d = np.sum(doc[1:]) # count non-pad
        c=model.test(doc)
        if n_d==0:
            continue
        else:
            cost.append(c/n_d)
    # print('The approximated perplexity for test set is: ',np.exp(np.mean(np.array(cost))))
    return np.exp(np.mean(np.array(cost)))

def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')