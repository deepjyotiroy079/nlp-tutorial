import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle   
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the LSTM model
model = load_model('hamlet_lstm_model.keras')
# print(model.summary())

# load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)



# predict
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]  # truncate to fit the model input
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


st.title("Hamlet Next Word Prediction")
st.write("Enter a text from Hamlet to predict the next word:")
user_input = st.text_area("Input Text", "To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1  # get the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, user_input, max_sequence_length)
    
    if next_word:
        st.write(f"Next word prediction: {next_word}")
    else:
        st.write("Could not predict the next word.")
else:
    st.write("Click the button to predict the next word based on your input.")