import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Keep these files in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "next_word_lstm.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer2.json"
CONFIG_PATH = BASE_DIR / "model_config2.json"


st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ Next Word Predictor")
st.write("Enter a phrase and let your trained LSTM predict what word may come next.")


@st.cache_resource
def load_assets():
    missing = [
        str(path.name)
        for path in (MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing file(s): " + ", ".join(missing)
            + ". Keep app.py, the model, tokenizer, and config in the same folder."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with TOKENIZER_PATH.open("r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return model, tokenizer, config


def get_config_value(config, possible_names, default=None):
    """Return the first matching config value, including common naming variants."""
    for name in possible_names:
        if name in config and config[name] is not None:
            return config[name]
    return default


try:
    model, tokenizer, config = load_assets()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info(
        "Check that the filenames in the folder match the filenames configured "
        "at the top of app.py. If you see a TensorFlow/Keras compatibility error, "
        "check the TensorFlow version used to train the model."
    )
    st.stop()

# Prefer the actual model input shape; fall back to the config if necessary.
model_input_length = None
try:
    if isinstance(model.input_shape, tuple) and model.input_shape[-1] is not None:
        model_input_length = int(model.input_shape[-1])
except (TypeError, ValueError):
    pass

configured_length = get_config_value(
    config,
    [
        "sequence_length",
        "seq_length",
        "max_sequence_length",
        "max_len",
        "SEQ_LENGTH",
        "sequence_len",
    ],
)

if model_input_length is not None:
    sequence_length = model_input_length
elif configured_length is not None:
    sequence_length = int(configured_length)
else:
    st.error(
        "I couldn't determine the input sequence length. Add a sequence_length "
        "field to model_config2.json or check the model's input shape."
    )
    st.stop()

output_shape = model.output_shape
if isinstance(output_shape, list):
    st.error("This app expects a model with one output tensor, but this model has multiple outputs.")
    st.stop()

try:
    output_vocab_size = int(output_shape[-1])
except (TypeError, ValueError):
    st.error("Could not determine the model's output vocabulary size.")
    st.stop()

# Keras Tokenizer uses integer IDs. With num_words set, IDs >= num_words
# are excluded by texts_to_sequences, so the output layer size is authoritative.
word_index = tokenizer.word_index
index_word = {int(index): word for word, index in word_index.items()}

oov_id = None
if tokenizer.oov_token:
    oov_id = word_index.get(tokenizer.oov_token)

st.caption(
    f"Model input length: {sequence_length} tokens · "
    f"Output classes: {output_vocab_size:,}"
)

with st.form("prediction_form"):
    user_text = st.text_area(
        "Enter your text",
        value="The meaning of life",
        placeholder="Type a few words...",
        height=120,
    )
    top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Predict next word", type="primary", use_container_width=True)

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    # Convert input words to the same integer IDs used during training.
    token_ids = tokenizer.texts_to_sequences([user_text])[0]

    if not token_ids:
        st.warning(
            "The tokenizer couldn't recognize any words in that input. "
            "Try a longer phrase using words likely to appear in the training data."
        )
        st.stop()

    # Keep the most recent context tokens and pad on the left, matching the
    # common Keras training setup for next-word prediction.
    token_ids = token_ids[-sequence_length:]
    padded = pad_sequences(
        [token_ids],
        maxlen=sequence_length,
        padding="pre",
        truncating="pre",
        value=0,
    )

    try:
        with st.spinner("Predicting..."):
            predictions = model.predict(padded, verbose=0)
        probabilities = np.asarray(predictions)[0].astype(float)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    if probabilities.ndim != 1:
        st.error("The model output is not a single probability vector; this app expects a softmax next-word classifier.")
        st.stop()

    # Only rank class IDs that exist in the model output layer.
    probabilities = probabilities[:output_vocab_size]
    candidate_ids = np.argsort(probabilities)[::-1]

    results = []
    for idx in candidate_ids:
        idx = int(idx)
        word = index_word.get(idx)

        # Ignore padding, unknown/unmapped classes, and the OOV placeholder.
        if idx == 0 or word is None:
            continue
        if oov_id is not None and idx == int(oov_id):
            continue

        results.append((word, float(probabilities[idx])))
        if len(results) >= top_k:
            break

    if not results:
        st.warning(
            "No displayable word predictions were found. Check that the tokenizer "
            "used here is the same tokenizer used when training the model."
        )
    else:
        st.subheader("Predicted next words")
        st.write(f"Context: **{user_text.strip()}**")

        for rank, (word, probability) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {word}** — {probability:.2%}")
            st.progress(max(0.0, min(1.0, probability)))

        st.caption(
            "Probabilities are the model's scores for its candidate words, not a guarantee "
            "that the next word will be correct."
        )

with st.expander("Model and config details"):
    st.write("Model file:", MODEL_PATH.name)
    st.write("Tokenizer file:", TOKENIZER_PATH.name)
    st.write("Config file:", CONFIG_PATH.name)
    st.json(config)
