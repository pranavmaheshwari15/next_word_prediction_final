import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# ---------------------------------------------------------------------
# Files: keep app.py, next_word_lstm.keras, tokenizer2.json and
# model_config2.json in the same folder.
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "next_word_lstm.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer2.json"
CONFIG_PATH = BASE_DIR / "model_config2.json"


st.set_page_config(
    page_title="NextWord | LSTM Text Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #748098;
        --line: #e6e9f2;
        --purple: #6657e8;
        --purple-dark: #5143cb;
        --soft-purple: #f0eeff;
        --surface: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(116, 96, 255, .11), transparent 33%),
            radial-gradient(ellipse at 90% 5%, rgba(81, 190, 220, .10), transparent 28%),
            #f7f8fc;
        color: var(--ink);
    }

    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .brand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2.1rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: .8rem;
        color: #172033;
        font-family: 'Manrope', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 2.65rem;
        height: 2.65rem;
        border-radius: 14px;
        color: white;
        font-size: 1.35rem;
        background: linear-gradient(135deg, #8477ff, #5143cb);
        box-shadow: 0 9px 22px rgba(102, 87, 232, .25);
    }
    .brand small {
        display: block;
        margin-top: .12rem;
        color: #8992a7;
        font-family: 'DM Sans', sans-serif;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .18em;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        padding: .55rem .85rem;
        border: 1px solid #e3e6ef;
        border-radius: 999px;
        background: rgba(255,255,255,.78);
        color: #69748b;
        font-size: .77rem;
        font-weight: 600;
    }
    .status-dot {
        width: .48rem;
        height: .48rem;
        border-radius: 50%;
        background: #19a77b;
        box-shadow: 0 0 0 4px rgba(25,167,123,.12);
    }

    .hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.6rem;
        padding: 2.3rem 2.4rem;
        border: 1px solid rgba(222,225,242,.9);
        border-radius: 26px;
        background: linear-gradient(115deg, rgba(255,255,255,.96), rgba(248,249,255,.88));
        box-shadow: 0 18px 50px rgba(32,42,80,.045);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -80px;
        top: -130px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(119,104,255,.18), transparent 68%);
        pointer-events: none;
    }
    .eyebrow {
        display: inline-block;
        margin-bottom: .85rem;
        padding: .4rem .7rem;
        border-radius: 999px;
        background: #f0eeff;
        color: #6657d9;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: #172033;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.55rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.06;
    }
    .hero h1 span {
        color: #6657e8;
        background: linear-gradient(90deg, #796aff, #41a9cb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        max-width: 700px;
        margin: 1rem 0 0;
        color: #778198;
        font-size: 1rem;
        line-height: 1.75;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e7e9f2 !important;
        border-radius: 20px !important;
        background: rgba(255,255,255,.92) !important;
        box-shadow: 0 12px 35px rgba(25,35,70,.035);
    }
    .section-kicker {
        margin: 0 0 .35rem;
        color: #8a93a8;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0;
        color: #1b2438;
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: -.035em;
    }
    .section-desc {
        margin: .35rem 0 1.15rem;
        color: #7b8499;
        font-size: .84rem;
        line-height: 1.55;
    }

    .stTextArea label, .stSelectbox label, .stSlider label {
        color: #39445a !important;
        font-weight: 700 !important;
        font-size: .84rem !important;
    }
    .stTextArea textarea {
        border: 1px solid #e2e5ef !important;
        border-radius: 14px !important;
        background: #fafbff !important;
        color: #202a40 !important;
        font-size: .98rem !important;
        line-height: 1.7 !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #8a7dff !important;
        box-shadow: 0 0 0 3px rgba(102,87,232,.11) !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 11px;
        border-color: #e2e5ef;
    }
    .stButton > button, .stFormSubmitButton > button {
        min-height: 2.75rem;
        border-radius: 11px;
        border: 1px solid #e2e5ef;
        font-weight: 700;
        transition: all .18s ease;
    }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
        border: none;
        color: white;
        background: linear-gradient(105deg, #7668f6, #5547d5);
        box-shadow: 0 8px 18px rgba(102,87,232,.2);
    }
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
        border: none;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 11px 22px rgba(102,87,232,.26);
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #a59cff;
        color: #5d50d5;
    }

    .result-hero {
        padding: 1.15rem;
        border: 1px solid #e9e7ff;
        border-radius: 16px;
        background: linear-gradient(135deg, #f6f5ff, #f6fbff);
    }
    .result-label {
        color: #7d82a1;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .result-word {
        margin: .25rem 0 .5rem;
        color: #5e50d6;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 3vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -.06em;
        overflow-wrap: anywhere;
    }
    .result-context {
        color: #56617a;
        font-size: .9rem;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }
    .result-context strong { color: #5e50d6; }
    .candidate {
        margin-top: .8rem;
        padding: .8rem .85rem;
        border: 1px solid #edf0f6;
        border-radius: 13px;
        background: #fff;
    }
    .candidate-top {
        display: flex;
        justify-content: space-between;
        gap: .75rem;
        align-items: center;
        margin-bottom: .45rem;
    }
    .candidate-word {
        color: #273148;
        font-size: .9rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .candidate-pct {
        color: #6657d9;
        font-size: .78rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .candidate-track {
        height: 6px;
        overflow: hidden;
        border-radius: 999px;
        background: #f0f1f8;
    }
    .candidate-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #8c7fff, #4db7d1);
    }
    .metric-card {
        padding: 1rem 1.05rem;
        border: 1px solid #e7e9f2;
        border-radius: 16px;
        background: rgba(255,255,255,.85);
    }
    .metric-label {
        color: #8891a5;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .metric-value {
        margin-top: .35rem;
        color: #1d2740;
        font-family: 'Manrope', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .metric-help {
        margin-top: .2rem;
        color: #818ba0;
        font-size: .75rem;
    }
    .empty-result {
        display: flex;
        min-height: 285px;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        border: 1px dashed #dfe3f0;
        border-radius: 16px;
        background: #fbfbff;
        text-align: center;
    }
    .empty-icon {
        display: grid;
        place-items: center;
        width: 3.5rem;
        height: 3.5rem;
        margin-bottom: .9rem;
        border-radius: 18px;
        background: #f0eeff;
        color: #6657e8;
        font-size: 1.6rem;
    }
    .empty-title {
        color: #2a344b;
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 800;
    }
    .empty-copy {
        max-width: 250px;
        margin-top: .4rem;
        color: #838ca1;
        font-size: .82rem;
        line-height: 1.6;
    }
    .fineprint {
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid #e6e9f2;
        color: #8992a7;
        font-size: .75rem;
        line-height: 1.6;
    }
    .stAlert { border-radius: 13px; }

    @media (max-width: 700px) {
        .block-container { padding: 1rem .85rem 2rem; }
        .brand-row { align-items: flex-start; flex-direction: column; margin-bottom: 1.25rem; }
        .hero { padding: 1.55rem 1.25rem; border-radius: 20px; }
        .hero h1 { letter-spacing: -.05em; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load assets once per Streamlit session.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading your trained LSTM model…")
def load_assets():
    missing = [
        path.name
        for path in (MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing file(s): " + ", ".join(missing)
            + ". Keep app.py, the model, tokenizer and config in the same folder."
        )

    # compile=False is appropriate for inference and avoids loading optimizer state.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with TOKENIZER_PATH.open("r", encoding="utf-8") as file:
        tokenizer = tokenizer_from_json(file.read())

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return model, tokenizer, config


def get_config_value(config, names, default=None):
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return default


def get_model_dimensions(model, config):
    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list) or isinstance(output_shape, list):
        raise ValueError("This app expects a single-input, single-output model.")

    input_length = input_shape[-1]
    if input_length is None:
        input_length = get_config_value(
            config,
            [
                "sequence_length", "seq_length", "max_sequence_length",
                "max_len", "SEQ_LENGTH", "sequence_len",
            ],
        )
    if input_length is None:
        raise ValueError("Could not determine the model's input sequence length.")

    output_vocab_size = output_shape[-1]
    if output_vocab_size is None:
        raise ValueError("Could not determine the model's output vocabulary size.")

    return int(input_length), int(output_vocab_size)


def escape_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def make_candidate_card(word, probability, rank, max_probability):
    # Relative bar lengths make alternatives easier to compare visually.
    relative = 0 if max_probability <= 0 else (probability / max_probability) * 100
    return f"""
    <div class="candidate">
      <div class="candidate-top">
        <span class="candidate-word"><span style="color:#a1a8ba;font-size:.72rem;margin-right:.45rem;">{rank:02d}</span>{escape_html(word)}</span>
        <span class="candidate-pct">{probability:.2%}</span>
      </div>
      <div class="candidate-track">
        <div class="candidate-fill" style="width:{relative:.2f}%;"></div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------
# Header and hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-row">
      <div class="brand">
        <div class="brand-icon">✦</div>
        <div>nextword<small>NEURAL TEXT STUDIO</small></div>
      </div>
      <div class="status-pill"><span class="status-dot"></span> LSTM inference workspace</div>
    </div>
    <section class="hero">
      <div class="eyebrow">✦ &nbsp; Powered by your trained model</div>
      <h1>Where will your sentence<br><span>go next?</span></h1>
      <p>Enter a phrase, run your LSTM model, and explore its most likely next-word predictions. Choose a suggestion to keep building your sentence.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load model and show a useful error instead of crashing the app.
# ---------------------------------------------------------------------
try:
    model, tokenizer, config = load_assets()
    sequence_length, output_vocab_size = get_model_dimensions(model, config)
    word_index = tokenizer.word_index
    index_word = {int(index): word for word, index in word_index.items()}
    oov_id = word_index.get(tokenizer.oov_token) if tokenizer.oov_token else None
except Exception as exc:
    st.error(f"Your model assets could not be loaded: {exc}")
    st.markdown(
        """
        **Check these files are together in the app folder:**
        - `app.py`
        - `next_word_lstm.keras`
        - `tokenizer2.json`
        - `model_config2.json`

        If the error mentions Keras serialization, confirm the installed Keras
        version matches the one used to save the model.
        """
    )
    st.stop()


if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "draft_text" not in st.session_state:
    st.session_state.draft_text = "The future of artificial intelligence"


# ---------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------
left, right = st.columns([1.12, 0.88], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">01 / Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your context</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">The model uses the recent words as context to estimate what may come next.</div>',
            unsafe_allow_html=True,
        )

        sample_col1, sample_col2, sample_col3 = st.columns(3)
        with sample_col1:
            if st.button("AI & tech", use_container_width=True):
                st.session_state.draft_text = "The future of artificial intelligence"
                st.rerun()
        with sample_col2:
            if st.button("Story starter", use_container_width=True):
                st.session_state.draft_text = "Once upon a time there was"
                st.rerun()
        with sample_col3:
            if st.button("Machine learning", use_container_width=True):
                st.session_state.draft_text = "Machine learning models can"
                st.rerun()

        with st.form("prediction_form", clear_on_submit=False):
            user_text = st.text_area(
                "Enter a phrase",
                key="draft_text",
                height=170,
                max_chars=1200,
                placeholder="Start typing your sentence here…",
                help="Use a phrase in the language and style represented in your training data.",
            )
            control_col1, control_col2 = st.columns([0.8, 1.2])
            with control_col1:
                top_k = st.select_slider(
                    "Suggestions to show",
                    options=[3, 5, 7, 10],
                    value=5,
                )
            with control_col2:
                st.markdown(
                    f'<div style="padding-top:2rem;color:#8a93a8;font-size:.78rem;text-align:right;">'
                    f'Context window: <strong style="color:#5d50d5;">{sequence_length} tokens</strong>'
                    f' &nbsp;·&nbsp; Vocabulary: <strong style="color:#5d50d5;">{output_vocab_size:,}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            submitted = st.form_submit_button(
                "✦  Predict next word",
                type="primary",
                use_container_width=True,
            )

        clear_col, hint_col = st.columns([0.55, 1.45])
        with clear_col:
            if st.button("Clear text", use_container_width=True):
                st.session_state.draft_text = ""
                st.session_state.last_results = []
                st.session_state.last_input = ""
                st.rerun()
        with hint_col:
            st.caption("Tip: try adding more context if the suggestions feel too broad.")


if submitted:
    text = user_text.strip()
    if not text:
        st.warning("Enter a phrase before predicting.")
    else:
        token_ids = tokenizer.texts_to_sequences([text])[0]
        if not token_ids:
            st.warning(
                "The tokenizer could not recognize any words in this phrase. "
                "Try words that are likely to appear in the model's training data."
            )
        else:
            token_ids = token_ids[-sequence_length:]
            padded = pad_sequences(
                [token_ids],
                maxlen=sequence_length,
                padding="pre",
                truncating="pre",
                value=0,
            )
            try:
                with st.spinner("Your model is thinking…"):
                    raw_predictions = model.predict(padded, verbose=0)
                probabilities = np.asarray(raw_predictions)[0].astype(float)

                if probabilities.ndim != 1:
                    raise ValueError("The model did not return one probability vector.")
                if len(probabilities) < output_vocab_size:
                    raise ValueError(
                        f"Model returned {len(probabilities)} classes, but the expected "
                        f"output vocabulary contains {output_vocab_size}."
                    )

                probabilities = probabilities[:output_vocab_size]
                # Ignore non-finite values and classes that cannot be mapped to words.
                probabilities = np.where(np.isfinite(probabilities), probabilities, 0.0)
                ranked_ids = np.argsort(probabilities)[::-1]

                results = []
                for token_id in ranked_ids:
                    token_id = int(token_id)
                    word = index_word.get(token_id)
                    if token_id == 0 or word is None:
                        continue
                    if oov_id is not None and token_id == int(oov_id):
                        continue
                    results.append((word, float(probabilities[token_id])))
                    if len(results) >= int(top_k):
                        break

                st.session_state.last_results = results
                st.session_state.last_input = text
                st.session_state.prediction_count += 1
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


with right:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">02 / Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction results</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">Your model’s highest-scoring next-word candidates.</div>',
            unsafe_allow_html=True,
        )

        results = st.session_state.last_results
        input_text = st.session_state.last_input

        if not results:
            st.markdown(
                """
                <div class="empty-result">
                  <div class="empty-icon">✧</div>
                  <div class="empty-title">Your results will appear here</div>
                  <div class="empty-copy">Enter a phrase on the left and run a prediction to see the model's actual output.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            best_word, best_probability = results[0]
            st.markdown(
                f"""
                <div class="result-hero">
                  <div class="result-label">Top prediction · #1</div>
                  <div class="result-word">{escape_html(best_word)}</div>
                  <div class="result-context">{escape_html(input_text)} <strong>{escape_html(best_word)}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="margin-top:1.35rem;margin-bottom:.3rem;color:#737e95;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;">Alternative predictions</div>',
                unsafe_allow_html=True,
            )
            max_probability = max((prob for _, prob in results), default=0.0)
            for rank, (word, probability) in enumerate(results, start=1):
                st.markdown(
                    make_candidate_card(word, probability, rank, max_probability),
                    unsafe_allow_html=True,
                )

            st.caption(
                "Scores are the model's output probabilities for these candidates—not a guarantee of correctness."
            )


# ---------------------------------------------------------------------
# Compact project metrics and technical details
# ---------------------------------------------------------------------
st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "ARCHITECTURE", "LSTM", "Sequential text model"),
    (m2, "INPUT WINDOW", str(sequence_length), "Recent token IDs"),
    (m3, "VOCABULARY", f"{output_vocab_size:,}", "Output classes"),
    (m4, "THIS SESSION", str(st.session_state.prediction_count), "Predictions run"),
]
for col, label, value, help_text in metrics:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Model details & files"):
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**Loaded assets**")
        st.write(f"Model: `{MODEL_PATH.name}`")
        st.write(f"Tokenizer: `{TOKENIZER_PATH.name}`")
        st.write(f"Config: `{CONFIG_PATH.name}`")
        st.markdown("**Model input shape**")
        st.code(str(model.input_shape))
        st.markdown("**Model output shape**")
        st.code(str(model.output_shape))
    with detail_right:
        st.markdown("**Configuration file**")
        st.json(config)
        st.markdown("**Runtime versions**")
        st.write(f"TensorFlow: `{tf.__version__}`")
        st.write(f"Keras: `{tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'version not exposed by tf.keras'}`")

st.markdown(
    """
    <div class="fineprint">
      <strong>About predictions:</strong> The app uses your trained model and its tokenizer.
      Output probabilities describe the model's relative scores, not certainty. Prediction quality
      depends on the training data, tokenization, and preprocessing matching the training pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)
