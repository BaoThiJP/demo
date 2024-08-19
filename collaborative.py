import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from gensim.similarities import Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
import time


# Tải mô hình từ file pickle
with open('baseline_model.pkl', 'rb') as file:
    model = pickle.load(file)
