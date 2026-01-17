import re
from typing import List
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from server.models.models import TextRequest

def preprocess_text(text: str) -> str:
    """
    Очистка текста: нижний регистр, удаление пунктуации и чисел.
    """
    text = text.lower()
    text = re.sub(r"[^а-яa-zё\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def handle_text_with_word2vec(request: TextRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")

    processed_texts: List[str] = [preprocess_text(text) for text in request.documents]
    
    vectorizer = TfidfVectorizer(
        max_features=1000,      # ограничение словаря
        min_df=1,               # минимальная частота слова
        max_df=0.95,            # игнорировать слишком частые слова
        token_pattern=r"[а-яa-zё]+"  # токенизация без цифр и знаков
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    word_doc_matrix = tfidf_matrix.T

    n_components = min(100, word_doc_matrix.shape[1], word_doc_matrix.shape[0])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    word_embeddings = svd.fit_transform(word_doc_matrix)

    vocab = vectorizer.get_feature_names_out().tolist()

    return {
        "vocabulary": vocab,
        "word_vectors": word_embeddings.tolist(),
        "shape": list(word_embeddings.shape)
    }