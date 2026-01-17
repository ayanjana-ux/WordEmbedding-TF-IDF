import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from gensim.models import Word2Vec

# DOWNLOAD NLTK DATA
nltk.download("punkt")
nltk.download("stopwords")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)

# APP TITLE
st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, and Word Embeddings")


# USER INPUT
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Word Embedding"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    # TOKENIZATION
    elif option == "Tokenization":
        st.subheader("Tokenization Output")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    # TEXT CLEANING
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

    # STEMMING
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)
        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    # LEMMATIZATION
    elif option == "Lemmatization":
        st.subheader("Lemmatization using spaCy")

        doc = nlp(text)
        df = pd.DataFrame(
            [(token.text, token.pos_, token.lemma_) for token in doc],
            columns=["Word", "POS", "Lemma"]
        )

        st.dataframe(df, use_container_width=True)

    # WORD EMBEDDING + REGEX
    elif option == "Word Embedding":
        st.subheader("Word Embedding using Word2Vec (Regex Based)")

        # REGEX TOKENIZATION (EXPLICIT)
        tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())

        if len(tokens) < 2:
            st.warning("Please enter more text for Word Embedding.")
        else:
            st.markdown("### Tokens after Regex Processing")
            st.write(tokens)

            # Train Word2Vec model
            model = Word2Vec(
                sentences=[tokens],
                vector_size=50,
                window=3,
                min_count=1,
                workers=1
            )

            # Create embedding table
            embedding_data = []
            for word in model.wv.index_to_key:
                embedding_data.append(
                    [word] + model.wv[word].tolist()
                )

            columns = ["Word"] + [f"Dim_{i+1}" for i in range(50)]
            df = pd.DataFrame(embedding_data, columns=columns)

            st.markdown("### Word Embedding Vectors")
            st.dataframe(df, use_container_width=True)
