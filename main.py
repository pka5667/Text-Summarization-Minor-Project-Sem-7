'''
Steps to be done:
- TODO: Neural Topic Model (Can be done using LDA)
- TODO: Word Embeddings
- TODO: Sentence Embeddings using CNN and BiLSTM on Word Embeddings
- TODO: Generate Graph (Topic, Sentance, Word)
- TODO: Apply Graph Attention Layer
- TODO: Create a sentance classifier
- TODO: Generate Summary after classifing sentances
'''

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer

# Importing the csv file
df = pd.read_csv('newEnglishNews_train.csv')

documents = df['articles']
summary = df['summary']

# Step: Neural Topic Model (with encoder and decoder model)

# Encoder 
encoder = LabelEncoder()
encoder.fit(summary)
encoded_summary = encoder.transform(summary)
