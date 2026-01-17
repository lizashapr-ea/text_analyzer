import numpy as np
import re
from math import log
from server.models.models import TextRequest


def preprocess_text(text: str) -> list:
    """
    Очистка текста:
    - в нижний регистр
    - удаление пунктуации и чисел
    - токенизация по пробелам
    """
    text = text.lower()
    text = re.sub(r"[^а-яa-zё\s]", " ", text)
    tokens = text.split()
    return tokens


def handle_text_with_tfidf(request: TextRequest) -> dict:
    """
    Реализация TF-IDF на numpy.
    Возвращает словарь с матрицей TF-IDF и словарём.
    """
    documents = request.documents
    tokenized_docs = [preprocess_text(doc) for doc in documents]

    vocabulary = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    tf_matrix = np.zeros((len(documents), len(vocabulary)))

    for doc_idx, tokens in enumerate(tokenized_docs):
        for token in tokens:
            if token in word_to_index:
                tf_matrix[doc_idx, word_to_index[token]] += 1

    tf_matrix = tf_matrix / np.maximum(tf_matrix.sum(axis=1, keepdims=True), 1)

    df = np.count_nonzero(tf_matrix > 0, axis=0)
    idf = np.log((len(documents)) / (df + 1)) + 1  # +1 для сглаживания

    tfidf_matrix = tf_matrix * idf

    return {
        "vocabulary": vocabulary,
        "matrix": tfidf_matrix.tolist()
    }
