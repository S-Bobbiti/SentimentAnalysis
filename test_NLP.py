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

# List of predefined stopwords in english language are obtained from nltk
  stop_words = set(stopwords.words('english'))
# Defining a function that removes stopwords 
  def remove_stopwords(input):
      return " ".join([word for word in str(input).split() if word not in stop_words])
    
# Maketrans is used for mapping the characters [7]
# And Punctuation is pre-initialized in string [8]
  def remove_punctuation(text):
      return text.translate(str.maketrans('', '', string.punctuation))

# 1. Load your saved model	
  model1=tf.keras.models.load_model('./models/Group12_NLP_model.h5')

# 2. Load your testing data
# Calling the function
  X_train,Y_train = load_from_data(f'./data/aclImdb/train',['neg','pos'])
  X_test,Y_test = load_from_data(f'./data/aclImdb/test',['neg','pos'])

# Data pre-processing
# Converting to lower case
  X_train = list(map(str.lower, X_train))
  X_test = list(map(str.lower, X_test))

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


# 3. Run prediction on the test data and print the test accuracy
# Calculating performance of the model on test set
  score = model1.evaluate(x_test, Y_test, verbose=0)
  print("Test accuracy: %.2f%%" %(score[1]*100))
  print("Test loss: %.2f%%" %(score[0]*100))