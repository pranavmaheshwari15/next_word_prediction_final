import json
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="📝",
    layout="centered"
)


# -----------------------------
# Load model and tokenizer once
# -----------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        "best_lstm_model.keras",
        compile=False
    )

    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    index_to_word = {
        int(index): word
        for word, index in tokenizer.word_index.items()
        if int(index) < config["vocab_size"]
    }

    return model, tokenizer, config, index_to_word


model, tokenizer, config, index_to_word = load_assets()

SEQ_LEN = int(config["seq_len"])
VOCAB_SIZE = int(config["vocab_size"])


# -----------------------------
# Prediction function
# -----------------------------
def predict_next_word(text, top_k=5):
    token_ids = tokenizer.texts_to_sequences([text])[0]

    if not token_ids:
        return []

    # Keep recent context
    token_ids = token_ids[-SEQ_LEN:]

    # RIGHT-padding: real tokens first, zeros afterwards
    padded = np.zeros((1, SEQ_LEN), dtype=np.int32)
    padded[0, :len(token_ids)] = token_ids

    # Run model inference
    probabilities = model(
        padded,
        training=False
    ).numpy()[0]

    # Padding is not a valid prediction
    probabilities[0] = -1.0

    top_ids = np.argsort(probabilities)[::-1]
    top_ids = [
        int(idx) for idx in top_ids
        if idx != 0 and idx < VOCAB_SIZE
    ][:top_k]

    results = []

    for idx in top_ids:
        word = index_to_word.get(idx, "<unk>")
        probability = float(probabilities[idx])

        results.append({
            "Predicted word": word,
            "Model probability": f"{probability:.2%}"
        })

    return results


# -----------------------------
# User interface
# -----------------------------
st.title("📝 LSTM Next-Word Predictor")

st.write(
    "Enter a sentence or phrase, and the trained LSTM "
    "will predict the most likely next words."
)

st.caption("Model trained on the WikiText-2 dataset.")

context = st.text_area(
    "Enter your text",
    value="the history of the united states",
    height=100,
    placeholder="Start typing here..."
)

top_k = st.slider(
    "Number of predictions",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Predict next word", type="primary"):
    if not context.strip():
        st.warning("Please enter some text.")
    else:
        predictions = predict_next_word(context, top_k)

        if predictions:
            st.subheader("Predictions")
            st.table(predictions)
        else:
            st.warning(
                "No recognized words found. Try another sentence."
            )

st.divider()
st.caption(
    "Probabilities are produced by the model and are not "
    "necessarily calibrated measures of real-world certainty."
)