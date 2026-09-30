import streamlit as st
import joblib
import re
import os


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="TruthCheck - Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "model",
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "model",
    "tfidf_vectorizer.pkl"
)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================
# HEADER
# =========================

st.title("📰 TruthCheck")

st.subheader("Fake News Detection")

st.write(
    "Check whether a news article is likely to be "
    "real or fake using Machine Learning."
)

st.divider()


# =========================
# INPUT
# =========================

title = st.text_input(
    "News Headline",
    placeholder="Enter the news headline..."
)

article = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=250
)


# =========================
# BUTTON
# =========================

if st.button(
    "🔍 Check News",
    use_container_width=True
):

    if not article.strip():

        st.warning(
            "Please enter a news article."
        )

    else:

        # Combine title and article
        content = clean_text(
            title + " " + article
        )

        # Convert text to TF-IDF
        vector = vectorizer.transform(
            [content]
        )

        # Prediction
        prediction = model.predict(
            vector
        )[0]

        # Probability
        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = max(
            probabilities
        ) * 100


        # =========================
        # RESULT
        # =========================

        if prediction == 1:

            st.success(
                "✅ Likely Real News"
            )

        else:

            st.error(
                "⚠️ Likely Fake News"
            )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


# =========================
# INFORMATION
# =========================

st.divider()

st.subheader("🤖 How It Works")

st.write(
    "This project uses TF-IDF for text feature extraction "
    "and Logistic Regression for classification."
)

st.info(
    "Note: The prediction is based on patterns learned "
    "from the training dataset and should not be treated "
    "as a definitive fact-check."
)


# =========================
# FOOTER
# =========================

st.caption(
    "TruthCheck • Fake News Detection • Machine Learning Project"
)