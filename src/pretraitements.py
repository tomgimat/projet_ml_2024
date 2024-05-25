# pretraitements.py

import numpy as np
import pandas as pd
import nltk
import sklearn
import sys
import os
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def nettoyer_texte(text):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


#Suppression de certains caracteres
def clean_text(text):
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\W+", " ", text)
    text = text.lower()
    return text


#Mise sous forme de liste
def tokenize(text):
    return word_tokenize(text)


# LEMMATISATION
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]