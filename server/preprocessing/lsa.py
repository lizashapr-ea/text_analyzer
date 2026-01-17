# server/preprocessing/lsa.py

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from server.models.models import TextRequest


def preprocess_text(text: str) -> str:
    """
    Очистка текста: нижний регистр, удаление пунктуации и чисел.
    Возвращает очищенную строку (для sklearn-векторизатора).
    """
    text = text.lower()
    text = re.sub(r"[^а-яa-zё\s]", " ", text)
    return text


def handle_text_with_lsa(request: TextRequest, n_components: int = 2) -> dict:
    """
    Реализация LSA через sklearn (TfidfVectorizer + TruncatedSVD)
    """
    documents = [preprocess_text(doc) for doc in request.documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    return {
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "reduced_matrix": reduced_matrix.tolist(),
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "n_components": n_components
    }
