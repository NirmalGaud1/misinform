import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load tokenizer from JSON
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()  # FIXED INDENTATION
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer  # FIXED: ADDED RETURN

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "bilstm_misinformation_model.h5",
        custom_objects={'LSTM': LSTM, 'Bidirectional': Bidirectional}
    )

# Preprocess input
def preprocess_text(text, tokenizer, maxlen=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
    return padded

# Main app
def main():
    st.title("🚩 Misinformation Detector using BiLSTM")
    st.write("Enter a tweet or sentence to check if it contains misinformation.")

    user_input = st.text_area("Enter text here:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            tokenizer = load_tokenizer()
            model = load_model()
            processed_input = preprocess_text(user_input, tokenizer)
            prediction = model.predict(processed_input)[0][0]
            label = "Misinformation ❌" if prediction >= 0.5 else "Not Misinformation ✅"
            st.success(f"Prediction: **{label}** (Confidence: {prediction:.2f})")

if __name__ == '__main__':
    main()
