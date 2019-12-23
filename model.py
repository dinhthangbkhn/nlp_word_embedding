import os

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Reshape, Embedding
from tensorflow.keras.models import Sequential
from word_embedding_corpus.py import vocab_size, embed_size, skip_grams
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size, input_length=1))
word_model.add(Reshape((embed_size,)))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,input_length=1))
context_model.add(Reshape((embed_size,)))

model = Sequential()
model.add(Concatenate([word_model, context_model], mode='dot'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
