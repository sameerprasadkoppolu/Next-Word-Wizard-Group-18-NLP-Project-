# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

# %%
with open('merged_articles.txt','r') as f:
    corpus = f.read()
training_doc1 = corpus.replace("\n",' ')

# %%
# data = pd.read_csv("/Users/maheshbabu/Desktop/NLP-Next-Word-Predictor/ArticlesJan2017.csv")
# corpus = data['snippet'].str.lower().tolist()

# training_doc1 = ''.join(corpus)

# %%


# %%
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

class MarkovChain:
    def __init__(self):
        self.lookup_dict = defaultdict(list)  

    def add_document(self, text):
        preprocessed_list = self._preprocess(text)
        pairs = self.__generate_tuple_keys(preprocessed_list)
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])
        pairs2 = self.__generate_2tuple_keys(preprocessed_list)
        for pair in pairs2:
            self.lookup_dict[tuple([pair[0], pair[1]])].append(pair[2])
        pairs3 = self.__generate_3tuple_keys(preprocessed_list)
        for pair in pairs3:
            self.lookup_dict[tuple([pair[0], pair[1], pair[2]])].append(pair[3])

    def _preprocess(self, text):
        cleaned = re.sub(r'\W+', ' ', text).lower()
        tokenized = word_tokenize(cleaned)
        return tokenized

    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return

        for i in range(len(data) - 1):
            yield [ data[i], data[i + 1] ]

    def __generate_2tuple_keys(self, data):
        if len(data) < 2:
            return

        for i in range(len(data) - 2):
            yield [ data[i], data[i + 1], data[i+2] ]


    def __generate_3tuple_keys(self, data):
        if len(data) < 3:
            return

        for i in range(len(data) - 3):
            yield [ data[i], data[i + 1], data[i+2], data[i+3] ]

    def suggest_one_word(self, word):
        return Counter(self.lookup_dict[word]).most_common()[:3]

    def suggest_two_words(self, words):
        suggestions = Counter(self.lookup_dict[tuple(words)]).most_common()[:3]
        if len(suggestions)==0:
            return self.suggest_one_word(words[-1])
        return suggestions

    def suggest_three_words(self, words):
        suggestions = Counter(self.lookup_dict[tuple(words)]).most_common()[:3]
        if len(suggestions)==0:
            return self.suggest_two_words(words[-2:])
        return suggestions

    def suggest_more_words(self, words):
        return self.suggest_three_words(words[-3:])

    def generate_text(self, text):
        if len(self.lookup_dict) > 0:
            tokens = text.split(" ")
            if len(tokens)==1:
                #print("Next word suggestions:", self.suggest_one_word(text))
                return self.suggest_one_word(text)
            elif len(tokens)==2:
                #print("Next word suggestions:", self.suggest_two_words(text.split(" ")))
                return self.suggest_two_words(text.split(" "))
            elif len(tokens)==3:
                #print("Next word suggestions:", self.suggest_three_words(text.split(" ")))
                return self.suggest_three_words(text.split(" "))
            elif len(tokens)>3:
                #print("Next word suggestions:", self.suggest_more_words(text.split(" ")))
                return self.suggest_more_words(text.split(" "))
        return

# %%
# while True:
#     my_markov = MarkovChain()
#     my_markov.add_document(training_doc1)
#     text = input("Enter the sentence to generate next word prediction: ").lower()
#     if text.strip() == '' or text.strip() == 'quit':
#         print("Testing is done")
#         break
#     else:
#         print("Possible next words are:")
#         my_markov.generate_text(text)
#         print("--"*30)

# %%



