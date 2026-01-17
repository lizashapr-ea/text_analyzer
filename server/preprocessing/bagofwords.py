import numpy as np
import re
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

def handle_text_with_bag_of_words(request: TextRequest) -> dict:
    """
    Преобразование списка документов в матрицу bag-of-words.
    Возвращает словарь с матрицей и словарём (вокабуляром).
    """
    documents = request.documents

    tokenized_docs = [preprocess_text(doc) for doc in documents]

    vocabulary = sorted(list(set(word for doc in tokenized_docs for word in doc)))

    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    matrix = np.zeros((len(documents), len(vocabulary)), dtype=int)

    for doc_idx, tokens in enumerate(tokenized_docs):
        for token in tokens:
            if token in word_to_index:
                matrix[doc_idx, word_to_index[token]] += 1

    return {
        "vocabulary": vocabulary,
        "matrix": matrix.tolist()
    }
