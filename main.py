import streamlit as st
import pandas as pd
import pickle
from MarkovChainModel import MarkovChain
from happytransformer import HappyWordPrediction

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras import layers

max_sequence_len = 37
with open('merged_articles.txt','r') as f:
    corpus = f.read()
training_doc1 = corpus.replace("\n",' ')

with open('tokenizer1.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.exp(K.mean(cross_entropy))
    return perplexity

# Create a dictionary of models and their corresponding predictions
models = {
    'Markov Chain Model': ['word1', 'word2', 'word3'],
    'LSTM': ['word4', 'word5', 'word6'],
    'BI-LSTM': ['word7', 'word8', 'word9'],
    'Distill-Bert Transformer' : ['word7', 'word8', 'word9'],
    'AlBert Transformer' : ['word7', 'word8', 'word9'],
    'Bert Transformer' : ['word7', 'word8', 'word9'],
    'RoBerta Transformer' : ['word7', 'word8', 'word9']
}

# Create a function to generate predictions based on selected model and input sentence
def generate_predictions(model, input_sentence):
    # In this example, we simply return the top 3 predictions for illustration purposes
    predictions = models[model][:3]
    return predictions

# Define custom CSS for aesthetics
CSS = """
h1 {
    font-size: 48px;
    color: #2C3E50;
    text-align: center;
}

.highlight {
    color: #FFFFFF;
    background-color: #3498DB;
    padding: 5px 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

label {
    font-size: 24px;
    color: #2C3E50;
}

select {
    font-size: 18px;
    padding: 5px 10px;
    margin-top: 5px;
    margin-bottom: 15px;
    border-radius: 5px;
    background-color: #F5F5F5;
    color: #333333;
}

input[type="text"] {
    font-size: 18px;
    padding: 5px 10px;
    margin-top: 5px;
    margin-bottom: 15px;
    border-radius: 5px;
    background-color: #F5F5F5;
    color: #333333;
}

table.dataframe {
    border-collapse: collapse;
    margin: 0 auto;
}

th {
    background-color: #3498DB;
    color: #FFFFFF;
    font-size: 24px;
    font-weight: normal;
    text-align: left;
    padding: 10px;
}

td {
    background-color: #F5F5F5;
    color: #333333;
    font-size: 18px;
    text-align: left;
    padding: 5px 10px;
    border-bottom: 1px solid #DDDDDD;
}
"""

# Add custom CSS for aesthetics
st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)

# Create a Streamlit app
st.title('Next Word Prediction Demo')
st.markdown('<h1>Try out different models for next word prediction!</h1>', unsafe_allow_html=True)
st.markdown('<div class="highlight"></div>', unsafe_allow_html=True)

# Create a dropdown menu for selecting a model
model = st.selectbox('Select a model', list(models.keys()))

# Markov Chain Model
if model == 'Markov Chain Model':
    input_sentence = st.text_input('Enter a sentence to get next word predictions',"What do you")
    my_markov = MarkovChain()
    my_markov.add_document(training_doc1)
    text = input_sentence.lower()
    if text.strip():
        st.write("Possible next words are:")
        result = my_markov.generate_text(text)
        predictions = [tup[0] for tup in result]
        df = pd.DataFrame({'Predictions': predictions})
        st.write(df)

elif model == 'LSTM':
    lstm_model = tf.keras.models.load_model('model_parameters_thismac/bilstm_model.h5',custom_objects={'perplexity': perplexity})
    next_words = 10
    input_sentence = st.text_input('Enter a sentence to get next word predictions',"What do you")
    seed_text = input_sentence
    st.write("The next possible sentences could be: ")
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = lstm_model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        st.write(seed_text)
    


elif model == 'BI-LSTM':
    bilstm_model = tf.keras.models.load_model('model_parameters_thismac/bilstm_model.h5',custom_objects={'perplexity': perplexity})
    next_words = 10
    input_sentence = st.text_input('Enter a sentence to get next word predictions',"Why are you")
    seed_text = input_sentence
    st.write("The next possible sentences could be: ")
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = bilstm_model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        st.write(seed_text)


# Distil Bert Transformer
elif model == 'Distill-Bert Transformer':
    happy_wp_distilbert = HappyWordPrediction(load_path="model/distilbert/")
    input_sentence = st.text_input("Enter the sentence by adding '[MASK]' token where you want to predict':\t",'What do you [MASK] to achieve?')
    if input_sentence.strip():
        text = input_sentence
        results = happy_wp_distilbert.predict_mask(text,top_k = 10)
        final_res = [(result.token , result.score) for result in results]
        df = pd.DataFrame(final_res, columns=['Possible Word', 'Probability'])
        st.write("Possible next words are: ")
        st.write(df)

# Albert Transformer
elif model == 'AlBert Transformer':
    happy_wp_albert = HappyWordPrediction(load_path="model/Albert/")
    input_sentence = st.text_input("Enter the sentence by adding '[MASK]' token where you want to predict':\t",'What do you [MASK] to achieve?')
    if input_sentence.strip():
        text = input_sentence
        results = happy_wp_albert.predict_mask(text,top_k = 10)
        final_res = [(result.token , result.score) for result in results]
        df = pd.DataFrame(final_res, columns=['Possible Word', 'Probability'])
        st.write("Possible next words are: ")
        st.write(df)

# Bert Transformer
elif model == 'Bert Transformer':
    happy_wp_bert = HappyWordPrediction(load_path="model/Bert/")
    input_sentence = st.text_input("Enter the sentence by adding '[MASK]' token where you want to predict':\t",'What do you [MASK] to achieve?')
    if input_sentence.strip():
        text = input_sentence
        results = happy_wp_bert.predict_mask(text,top_k = 10)
        final_res = [(result.token , result.score) for result in results]
        df = pd.DataFrame(final_res, columns=['Possible Word', 'Probability'])
        st.write("Possible next words are: ")
        st.write(df)

# Roberta Transformer
elif model == 'RoBerta Transformer':
    happy_wp_roberta = HappyWordPrediction(load_path="model/Roberta/")
    input_sentence = st.text_input("Enter the sentence by adding '<mask>' token where you want to predict':\t",'What do you <mask> to achieve?')
    if input_sentence.strip():
        text = input_sentence
        results = happy_wp_roberta.predict_mask(text,top_k = 10)
        final_res = [(result.token , result.score) for result in results]
        df = pd.DataFrame(final_res, columns=['Possible Word', 'Probability'])
        st.write("Possible next words are: ")
        st.write(df)



