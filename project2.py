import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize


stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


#################### App Making ###########################
st.title("Word Embedding with Filtering & Regularization")
st.write("Enter multiple sentences (one per line):")
text_input = st.text_area("Text Input")

if st.button("Generate Word Embeddings"):
    
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentences = text_input.split("\n")
        processed_sentences = [preprocess(sentence) for sentence in sentences]
        model = Word2Vec(
            sentences=processed_sentences,
            vector_size=50,
            window=3,
            min_count=1,
            workers=2
        )
        words = list(model.wv.index_to_key)
        vectors = np.array([model.wv[word] for word in words])
        vectors_normalized = normalize(vectors)

        df = pd.DataFrame(vectors_normalized, index=words)

        st.subheader("Normalized Word Embeddings (L2 Regularized)")
        st.dataframe(df)