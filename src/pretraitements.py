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


#Suppression de certains caracteres
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


#Mise sous forme de liste
def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


# LEMMATISATION
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


#VECTORISATION TF-IDF 
def vectorize_text_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)


#Appel des fonctions sur l'ensemble des textes
def complete_preprocess(texts):
    clean_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize(text) for text in clean_texts]
    filtered_texts = [remove_stopwords(tokens) for tokens in tokenized_texts]
    lemmatized_texts = [" ".join(lemmatize_tokens(tokens)) for tokens in filtered_texts]
    vectorized_texts = vectorize_text_tfidf(lemmatized_texts)
    return vectorized_texts