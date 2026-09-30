import pandas as pd
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==============================
# GET CORRECT PROJECT PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

fake_path = os.path.join(
    DATASET_DIR,
    "Fake.csv"
)

true_path = os.path.join(
    DATASET_DIR,
    "True.csv"
)

if not os.path.exists(fake_path) or not os.path.exists(true_path):
    raise FileNotFoundError(
        f"Dataset files not found. Checked: {DATASET_DIR}"
    )


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(csv_path, label_name, fallback_rows):
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        if not df.empty:
            return df

    print(f"{label_name} dataset is empty or missing. Creating fallback sample rows...")
    return pd.DataFrame(fallback_rows)


print("Loading Fake.csv...")
fake = load_dataset(
    fake_path,
    "Fake",
    [
        {"title": "Breaking: local market closes early", "text": "Authorities say the market closed due to a power outage and residents are confused about the sudden announcement."},
        {"title": "Shock video claims aliens landed in city", "text": "A viral hoax video shows a bright object in the sky but experts say there is no evidence of extraterrestrial activity."},
        {"title": "City officials deny new tax proposal", "text": "Officials released a statement saying no new tax will be introduced and that social media rumors are false."},
        {"title": "Doctor warns about fake miracle cure", "text": "Medical experts debunk a viral treatment and urge people not to trust unsupported health claims online."},
        {"title": "Emergency alert was a prank, police say", "text": "Police confirmed the emergency text was fabricated and reminded residents to verify before sharing alarming messages."}
    ]
)

print("Loading True.csv...")
true = load_dataset(
    true_path,
    "True",
    [
        {"title": "New park opens after renovation", "text": "City leaders officially opened the renovated park on Saturday after months of repairs and community improvements."},
        {"title": "Scientists report progress on clean energy", "text": "Researchers published findings showing improved battery efficiency and lower production costs for a new clean energy model."},
        {"title": "School district launches literacy program", "text": "The district announced a new reading initiative designed to improve student performance across elementary schools."},
        {"title": "Weather service confirms mild weekend forecast", "text": "Meteorologists said the weekend would remain mostly dry with cooler temperatures and low winds across the region."},
        {"title": "Hospital opens new cancer center", "text": "The regional hospital celebrated the opening of a specialized cancer treatment center with local officials and medical staff in attendance."}
    ]
)


# ==============================
# ADD LABELS
# ==============================

fake["label"] = 0
true["label"] = 1


# ==============================
# COMBINE DATA
# ==============================

df = pd.concat(
    [fake, true],
    ignore_index=True
)


# ==============================
# REMOVE EMPTY TEXT
# ==============================

df = df.dropna(
    subset=["text"]
)


# ==============================
# COMBINE TITLE + ARTICLE
# ==============================

df["content"] = (
    df["title"].fillna("") +
    " " +
    df["text"].fillna("")
)


# ==============================
# TEXT CLEANING
# ==============================

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


df["content"] = df["content"].apply(
    clean_text
)


# ==============================
# FEATURES AND LABEL
# ==============================

X = df["content"]
y = df["label"]


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# TF-IDF
# ==============================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


# ==============================
# TRAIN MODEL
# ==============================

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==============================
# TEST MODEL
# ==============================

predictions = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("==============================")
print("MODEL TRAINED SUCCESSFULLY!")
print("==============================")
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==============================
# CREATE MODEL FOLDER
# ==============================

model_folder = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(
    model_folder,
    exist_ok=True
)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    model,
    os.path.join(
        model_folder,
        "fake_news_model.pkl"
    )
)

joblib.dump(
    vectorizer,
    os.path.join(
        model_folder,
        "tfidf_vectorizer.pkl"
    )
)


print()
print("Model files saved successfully!")
print()
print("Created:")
print("model/fake_news_model.pkl")
print("model/tfidf_vectorizer.pkl")