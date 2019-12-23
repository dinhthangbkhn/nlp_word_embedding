import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '9'

import numpy as np
import nltk
import re
from nltk.corpus import gutenberg
from string import punctuation

#def normalize_corpus(doc):
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]','', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    #print(doc)
    return doc

#if __name__ == '__main__':
def get_data():
    bible = gutenberg.sents('bible-kjv.txt')
    remove_terms = punctuation + '0123456789'

    norm_bible = [[word.lower() for word in sent if word not in remove_terms]
                  for sent in bible]
    norm_bible = [' '.join(token_sent) for token_sent in norm_bible]
    print('befor filter', norm_bible[0])
    normalize_corpus = np.vectorize(normalize_document)
    norm_bible = normalize_corpus(norm_bible)
    print('After filtere', norm_bible[0])
    norm_bible = [token_sent for token_sent in norm_bible if len(token_sent.split())>2]
    print('Total lines', len(bible))
    print('sample line', bible[10])
    print('processed_line', norm_bible[10])
    return norm_bible

# build skip gram model
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.preprocessing import text


norm_bible = get_data()
# build vocabulary
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)

word2id = tokenizer.word_index
id2word = {v:k for k,v in word2id.items()}

vocab_size = len(word2id)+1
embed_size = 100

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])

# build and view sample skip grams
from tensorflow.keras.preprocessing.sequence import skipgrams

skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=5)
              for wid in wids]
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print('{:s}({:d}), {:s}({:d})'.format(
        id2word[pairs[i][0]], pairs[i][0],
        id2word[pairs[i][1]], pairs[i][1],
        labels[i]
    ))


from tensorflow.keras.layers import Input, Dot, Concatenate, Dense, Reshape, Embedding
from tensorflow.keras.models import Sequential, Model
def build_model():
    input_target = Input((1,))
    input_context = Input((1,))
    embedding = Embedding(vocab_size, embed_size)
    target = Embedding(vocab_size, embed_size)(input_target)
    target = Reshape((embed_size, 1))(target)
    context = Embedding(vocab_size, embed_size)(input_context)
    context = Reshape((embed_size, 1))(context)
    similarity = Dot(axes=1)([target, context])
    similarity = Reshape((1,))(similarity)
    output = Dense(1, activation='sigmoid')(similarity)
    model = Model(inputs=[input_target, input_context], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

model = build_model()
print(model.summary())
# train model
for epoch in range(1, 6):
    loss = 0
    for i, elem in enumerate(skip_grams):
        pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        Y = labels
        if i % 10000 == 0:
            print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
        loss += model.train_on_batch(X,Y)

    print('Epoch:', epoch, 'Loss:', loss)
model.save('logs/my_model.h5')



