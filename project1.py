import streamlit as st
import nltk
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

nltk.download("punkt")



def SplitCorpus(corpus):
    if isinstance(corpus, list):
        return corpus
    else:
        return sent_tokenize(corpus)


def tfIdf(corpus, query):
    corpus = SplitCorpus(corpus)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])

    similarity_scores = cosine_similarity(
        tfidf_matrix[-1], tfidf_matrix[:-1]
    ).flatten()

    result = pd.DataFrame({
        "Sentence": corpus,
        "TF-IDF Score": similarity_scores
    }).sort_values(by="TF-IDF Score", ascending=False)

    return result



def tokenize(text):
    raw_sentences = sent_tokenize(text)
    sentences = [word_tokenize(sentence.lower()) for sentence in raw_sentences]
    return sentences


def cbow_model(sentences, target, compare_word):
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        sg=0
    )

    if target in model.wv and compare_word in model.wv:
        similarity = model.wv.similarity(target, compare_word)
        return similarity
    return None


def skipgram_model(sentences, target, compare_word):
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        sg=1
    )

    if target in model.wv and compare_word in model.wv:
        similarity = model.wv.similarity(target, compare_word)
        return similarity
    return None



st.set_page_config(page_title="NLP App", layout="wide")
st.title("NLP Mini Website")
st.write("TF-IDF Text Filtering & Word Embedding Demo")


option = st.sidebar.radio(
    "Select Feature",
    ["TF-IDF", "Word Embedding"]
)


if option == "TF-IDF":
    st.header(" TF-IDF Text Filtering")

    text = st.text_area(
        "Enter Paragraph",
        height=150,
        placeholder="Enter Paragraph."
    )

    query = st.text_input(
        "Enter Query",
        placeholder="Enter Query."
    )

    if st.button("Run TF-IDF"):
        if text and query:
            result = tfIdf(text, query)
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("Please enter both text and query.")




elif option == "Word Embedding":
    st.header("Word Embedding (CBOW & Skip-gram)")

    text_input = st.text_area(
        "Enter Paragraph",
        height=150,
        placeholder="Enter Paragraph."
    )

    target_word = st.text_input("Target Word (Embedding Word)").lower()
    compare_word = st.text_input("Compare With").lower()

    if st.button("Run Word Embedding"):
        if text_input and target_word and compare_word:
            sentences = tokenize(text_input)

            st.subheader("Tokenized Sentences")
            st.write(sentences)

            cbow_sim = cbow_model(sentences, target_word, compare_word)
            skip_sim = skipgram_model(sentences, target_word, compare_word)

            if cbow_sim is not None:
                st.success(f"CBOW Similarity ({target_word}, {compare_word}) : {cbow_sim:.4f}")
            else:
                st.error("CBOW: Word not found in vocabulary")

            if skip_sim is not None:
                st.success(f"Skip-gram Similarity ({target_word}, {compare_word}) : {skip_sim:.4f}")
            else:
                st.error("Skip-gram: Word not found in vocabulary")
        else:
            st.warning("Please fill all inputs.")