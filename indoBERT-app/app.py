import streamlit as st
import joblib
import re
import os
import numpy as np

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 84px;
    font-weight: bold;
    color: #111827;
    line-height: 1.0;
    margin-bottom:10px;
}

.subtitle {
    text-align: center;
    font-size: 24px;
    color: #6b7280;
    margin-bottom: 50px;
    font-weight: 400;
}

.stTextArea textarea {
    border-radius: 15px;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    background-color: #4f46e5;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    background-color: #4338ca;
    color: white;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.positive {
    background-color: #dcfce7;
    color: #166534;
}

.neutral {
    background-color: #fef3c7;
    color: #92400e;
}

.negative {
    background-color: #fee2e2;
    color: #991b1b;
}

.confidence-label {
    text-align: center;
    font-size: 16px;
    color: #4b5563;
    margin-top: 8px;
    margin-bottom: 20px;
}

.confidence-section {
    margin-top: 25px;
}

.confidence-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}

.confidence-name {
    width: 90px;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
}

.confidence-bar-bg {
    flex-grow: 1;
    background-color: #e5e7eb;
    border-radius: 8px;
    height: 18px;
    margin: 0 10px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 8px;
}

.confidence-percent {
    width: 55px;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    text-align: right;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# BASE DIRECTORY
# =====================================

BASE_DIR = os.path.dirname(__file__)

# =====================================
# LOAD MODEL
# =====================================

model_path = os.path.join(
    BASE_DIR,
    "model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)

# =====================================
# STOPWORDS
# =====================================

factory = StopWordRemoverFactory()

stopwords = set(
    factory.get_stop_words()
)

# =====================================
# LABEL MAPPING
# =====================================
# Catatan: model.classes_ pada LinearSVC berisi label string
# ("negative", "neutral", "positive") karena dilatih langsung
# dengan y berupa string (lihat df['sentiment']), bukan id integer.
# id2label disiapkan untuk jaga-jaga jika suatu saat label berupa angka.

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Warna tiap kelas sentimen untuk bar confidence
sentiment_colors = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444"
}

# =====================================
# PREPROCESSING
# =====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    tokens = text.split()

    filtered_tokens = [
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(filtered_tokens)


def softmax(scores):
    """
    Mengubah skor decision_function (jarak ke hyperplane)
    menjadi confidence score berbentuk probabilitas (0-1)
    yang berjumlah 1 antar kelas.

    LinearSVC tidak punya predict_proba(), jadi confidence
    di sini dihitung dari softmax atas decision_function(),
    bukan probabilitas kalibrasi sebenarnya. Ini adalah
    pendekatan umum yang dipakai untuk linear SVM agar tetap
    bisa menampilkan tingkat keyakinan model per kelas.
    """
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header("📌 About Project")

    st.write("""
    Aplikasi ini menggunakan:
    
    - Deep Learning (IndoBERT)
    - Streamlit Deployment
    
    Untuk melakukan analisis sentimen review Tokopedia.
    """)

    st.info("Created by Emmanuel Alexander")

# =====================================
# MAIN TITLE
# =====================================

st.markdown(
    '<p class="title">🧠 Sentiment Analysis</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Analisis Sentimen Review Tokopedia menggunakan Deep Learning</p>',
    unsafe_allow_html=True
)

# =====================================
# INPUT
# =====================================

user_input = st.text_area(
    "✍️ Masukkan Review",
    height=180,
    placeholder="Contoh: Produk sangat bagus dan pengiriman cepat..."
)

# =====================================
# BUTTON
# =====================================

if st.button("🔍 Predict Sentiment"):

    if user_input.strip() != "":

        with st.spinner("Menganalisis sentimen..."):

            # preprocessing
            cleaned_text = clean_text(
                user_input
            )

            final_text = remove_stopwords(
                cleaned_text
            )

            # transform
            transformed_text = vectorizer.transform(
                [final_text]
            )

            # decision_function -> skor mentah per kelas (LinearSVC)
            decision_scores = model.decision_function(
                transformed_text
            )[0]

            # urutan kelas sesuai model.classes_
            class_labels = model.classes_

            # softmax untuk dapat confidence score (0-1) per kelas
            confidence_scores = softmax(decision_scores)

            # kelas dengan confidence tertinggi = hasil prediksi
            # (hasilnya akan selalu sama dengan model.predict())
            best_idx = int(np.argmax(confidence_scores))
            prediction = class_labels[best_idx]
            confidence = confidence_scores[best_idx]

            # mapping confidence per label, untuk ditampilkan semua
            confidence_per_label = {
                class_labels[i]: confidence_scores[i]
                for i in range(len(class_labels))
            }

            # label handling (jaga-jaga kalau prediction berupa angka)
            if prediction in id2label:
                sentiment = id2label[prediction]
            else:
                sentiment = str(prediction)

        # =====================================
        # RESULT UI
        # =====================================

        if sentiment == "positive":
            emoji = "😊"
            result_class = "positive"
        elif sentiment == "neutral":
            emoji = "😐"
            result_class = "neutral"
        else:
            emoji = "😡"
            result_class = "negative"

        st.markdown(
            f'''
            <div class="result-box {result_class}">
                {emoji} Hasil Sentiment: {sentiment.upper()}
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            f'''
            <div class="confidence-label">
                Confidence Score: <b>{confidence * 100:.2f}%</b>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Breakdown confidence untuk semua kelas
        bars_html = '<div class="confidence-section">'

        # urutkan tampilan: positive, neutral, negative
        display_order = ["positive", "neutral", "negative"]

        for label in display_order:
            if label not in confidence_per_label:
                continue

            pct = confidence_per_label[label] * 100
            color = sentiment_colors.get(label, "#6366f1")

            bars_html += f'''
            <div class="confidence-row">
                <div class="confidence-name">{label.capitalize()}</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{pct:.2f}%; background-color:{color};"></div>
                </div>
                <div class="confidence-percent">{pct:.1f}%</div>
            </div>
            '''

        bars_html += '</div>'

        st.markdown(bars_html, unsafe_allow_html=True)

    else:

        st.warning(
            "⚠️ Masukkan teks terlebih dahulu"
        )

# =====================================
# FOOTER
# =====================================

st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit</div>',
    unsafe_allow_html=True
)
