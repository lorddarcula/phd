# to hide warnings
import warnings
warnings.filterwarnings('ignore')

# basic data processing
import os
import datetime
import pandas as pd
import numpy as np

# for EDA
from pandas_profiling import ProfileReport

# for text preprocessing
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from spellchecker import SpellChecker

# progress bar
from tqdm.auto import tqdm
from tqdm import tqdm_notebook

# instantiate
tqdm.pandas(tqdm_notebook)

# for wordcloud
from PIL import Image
from wordcloud import WordCloud

# for aesthetics and plots
from IPython.display import display, Markdown, clear_output
from termcolor import colored

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.offline import plot, iplot
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"

# for model
import tensorflow as tf
import tensorflow_hub as hub
import keras.layers as layers
from keras.models import Model
from keras import backend as K
import keras
from keras.models import load_model

display(Markdown('_All libraries are imported successfully!_'))


col_names =  ['target', 'id', 'date', 'flag','user','text']

df = pd.read_csv('/rahul/projects/sentimentanalysis/training.1000000.processed.noemoticon.csv', encoding = "ISO-8859-1", names=col_names)

print(colored('DATA','blue',attrs=['bold']))
display(df.head())

# dropping irrelevant columns
df.drop(['id', 'date', 'flag', 'user'], axis=1, inplace=True)

# replacing positive sentiment 4 with 1
df.target = df.target.replace(4,1)

target_count = df.target.value_counts()

category_counts = len(target_count)
display(Markdown('__Number of categories__: {}'.format(category_counts)))



# set of stop words declared
stop_words = stopwords.words('english')

display(Markdown('__List of stop words__:'))
display(Markdown(str(stop_words)))


updated_stop_words = stop_words.copy()
for word in stop_words:
    if "n't" in word or "no" in word or word.endswith('dn') or word.endswith('sn') or word.endswith('tn'):
        updated_stop_words.remove(word)

# custom select words you don't want to eliminate
words_to_remove = ['for','by','with','against','shan','don','aren','haven','weren','until','ain','but','off','out']
for word in words_to_remove:
    updated_stop_words.remove(word)

display(Markdown('__Updated list of stop words__:'))
display(Markdown(str(updated_stop_words)))


# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# creating instance of spellchecker
spell = SpellChecker()

# creating instance of lemmatizer
lemm = WordNetLemmatizer()


def preprocess(tweet):
    # lowercase the tweets
    tweet = tweet.lower().strip()
    
    # REMOVE all URls
    tweet = re.sub(urlPattern,'',tweet)
    
    # Replace all emojis.
    for emoji in emojis.keys():
        tweet = tweet.replace(emoji, "emoji" + emojis[emoji])        
    
    # Remove @USERNAME
    tweet = re.sub(userPattern,'', tweet)        
    
    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " ", tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    splitted_tweet = tweet.split()
    # spell checks
#     misspelled = spell.unknown(splitted_tweet)
#     if misspelled == set():
#         pass
#     else:
#         for i,word in enumerate(misspelled):
#             splitted_tweet[i] = spell.correction(word)

    tweetwords = ''
    for word in splitted_tweet:
        # Checking if the word is a stopword.
        if word not in updated_stop_words:
            if len(word)>1:
                # Lemmatizing the word.
                lem_word = lemm.lemmatize(word)
                tweetwords += (lem_word+' ')
    
    return tweetwords
  
  df['text'] = df['text'].progress_apply(lambda x: preprocess(x))
print(colored('DATA','blue',attrs=['bold']))
display(df.head())
