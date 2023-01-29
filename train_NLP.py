# import required packages
# List of libraries used:
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
import tarfile
from glob import glob
import os
import re
import string
import nltk
# To use nltk tokenizer punkt is required
nltk.download('punkt')
# To use stopwords it needs to be downloaded from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from random import sample
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

if __name__ == "__main__": 

# Downloading data from stanford database [1].
  data_files = tf.keras.utils.get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', untar=False, md5_hash=None, cache_subdir='datasets', hash_algorithm='auto',extract=False, archive_format='auto', cache_dir=None)
# Tarfile is used to extract data from the zipfiles
  tar_file = tarfile.open(data_files)
# Data files are extracted into 'data' folder
  tar_file.extractall('./data/') 
  tar_file.close()

  def load_from_data(path,folder):
    text,labels = [],[]
    for i,label in enumerate(folder):
    # The following loop recursively creates a list containing all data contained in path
      for fname in glob(os.path.join(path, label, '*.*')):
       text.append(open(fname, 'r').read())
       labels.append(i)
    # lables are stored as numpy arrays
    return text, np.array(labels).astype(np.int64)
  
# Calling the function
  X_train,Y_train = load_from_data(f'./data/aclImdb/train',['neg','pos'])
  X_test,Y_test = load_from_data(f'./data/aclImdb/test',['neg','pos'])

# Data pre-processing
# Converting to lower case
  X_train = list(map(str.lower, X_train))
  X_test = list(map(str.lower, X_test))

# List of predefined stopwords in english language are obtained from nltk
  stop_words = set(stopwords.words('english'))
# Defining a function that removes stopwords 
  def remove_stopwords(input):
      return " ".join([word for word in str(input).split() if word not in stop_words])
    
# Maketrans is used for mapping the characters [7]
# And Punctuation is pre-initialized in string [8]
  def remove_punctuation(text):
      return text.translate(str.maketrans('', '', string.punctuation))

  X_train_stop = list(map(remove_stopwords, X_train))
  X_train_punc = list(map(remove_punctuation, X_train_stop))

  X_test_stop = list(map(remove_stopwords, X_test))
  X_test_punc = list(map(remove_punctuation, X_test_stop))

# Tokenizing
  tok = keras.preprocessing.text.Tokenizer()
# fit_on_text creates the vocabulary index based on word frequency
  tok.fit_on_texts(X_train_stop) 
# text_to_sequences assigns each sentence text into a sequence of integers
  X_train_tok = tok.texts_to_sequences(X_train_punc)
  X_test_tok = tok.texts_to_sequences(X_test_punc)

  x_train = X_train_tok
  x_test = X_test_tok

# Padding is done by appending 0's after the sentence words for a specified length
  max_length = 1500
  x_train = pad_sequences(x_train, padding = 'post', maxlen = max_length)
  x_test = pad_sequences(x_test, padding = 'post', maxlen = max_length)

# Splitting the training dataset into train (90%) and validation (10%)
  X_train, X_val, y_train, y_val = train_test_split(x_train, Y_train, test_size=0.2, random_state=27)

# Building Model
# Using CNN
  model1 = keras.Sequential(
      [
     # input dim is the vocabulary count used for the movie reviews (over 89,500 words)
          keras.layers.Embedding(input_dim=90000, output_dim=32, input_length = max_length),
          keras.layers.Dropout(0.2),
          keras.layers.Conv1D(32, kernel_size=(2), padding="same", activation="relu"),
          keras.layers.Dropout(0.2),
          keras.layers.Conv1D(32, kernel_size=(2), padding="same", activation="relu"),
          keras.layers.GlobalAveragePooling1D(),
          keras.layers.Dropout(0.2),
          keras.layers.Dense(1, activation="sigmoid"),
      ]
  )
  model1.summary()

  model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training the model
  history = model1.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=20,verbose=1,batch_size=512)

# Printing final training and validation accuracies and losses
  train_loss = model1.history.history['loss']
  val_loss   = model1.history.history['val_loss']
  train_acc  = model1.history.history['accuracy']
  val_acc    = model1.history.history['val_accuracy']

  print("Training loss at 20th epoch: ",train_loss[19:20])
  print("Training accuracy at 20th epoch: ",train_acc[19:20])
  print("Validation accuracy at 20th epoch: ",val_acc[19:20])
  print("Validation loss at 20th epoch: ",val_loss[19:20])

# Defining functions to plot the accuracy and loss with respesct to epoch
  def accuracy_plot(history):  
    plt.plot(history.history["accuracy"],label="train accuracy") 
    plt.plot(history.history["val_accuracy"],label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid()
    plt.show()

  def loss_plot(history): 
    plt.plot(history.history["loss"],label="train loss") 
    plt.plot(history.history["val_loss"],label="val loss") 
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid()
    plt.show()
  
  accuracy_plot(history)
  loss_plot(history)


# Saving the model for testing purpose
  model1.save('./models/Group12_NLP_model.h5')
