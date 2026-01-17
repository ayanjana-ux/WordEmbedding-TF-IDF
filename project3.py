import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("punkt")
nltk.download("stopwords")


nlp = spacy.load("en_core_web_sm")


st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)


st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, and TF-IDF")


text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "TF-IDF"
    ]
)

if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    elif option == "Tokenization":
        st.subheader("Tokenization Output")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            sentences = sent_tokenize(text)
            st.write(sentences)

        with col2:
            st.markdown("### Word Tokenization")
            words = word_tokenize(text)
            st.write(words)

        with col3:
            st.markdown("### Character Tokenization")
            characters = list(text)
            st.write(characters)

    elif option == "Text Cleaning":
        st.subheader("Text Cleaning Output")

        text_lower = text.lower()

        cleaned_text = "".join(
            ch for ch in text_lower
            if ch not in string.punctuation and not ch.isdigit()
        )

        doc = nlp(cleaned_text)
        final_words = [
            token.text for token in doc
            if not token.is_stop and token.text.strip() != ""
        ]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        porter_stem = [porter.stem(word) for word in words]
        lancaster_stem = [lancaster.stem(word) for word in words]

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": porter_stem,
            "Lancaster Stemmer": lancaster_stem
        })

        st.dataframe(df, use_container_width=True)

    elif option == "Lemmatization":
        st.subheader("Lemmatization using spaCy")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation (Regex Based)")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"[a-zA-Z]{2,}"
        )

        X = vectorizer.fit_transform([text])

        vocab = vectorizer.get_feature_names_out()
        tfidf_scores = X.toarray()[0]

        df = pd.DataFrame({
            "Word": vocab,
            "TF-IDF Score": tfidf_scores
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.markdown("### TF-IDF Table")
        st.dataframe(df, use_container_width=True)

        st.markdown("### TF-IDF Score Distribution (Top 10)")

        top_n = 10
        df_top = df.head(top_n)

        fig, ax = plt.subplots()
        ax.bar(df_top["Word"], df_top["TF-IDF Score"])
        ax.set_xlabel("Words")
        ax.set_ylabel("TF-IDF Score")
        plt.xticks(rotation=45)

        st.pyplot(fig)
