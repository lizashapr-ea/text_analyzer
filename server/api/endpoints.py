from fastapi import FastAPI
from server.models.models import TextRequest
from server.preprocessing.bagofwords import handle_text_with_bag_of_words
from server.preprocessing.textnltk import process_text_request
from server.preprocessing.tfidf import handle_text_with_tfidf
from server.preprocessing.lsa import handle_text_with_lsa
from server.preprocessing.word2vec import handle_text_with_word2vec

app = FastAPI()

@app.get("/")
async def root():
    '''Корневой эндпоинт'''
    return {"message": "Hello, FastAPI !!!"}

@app.post("/bagofwords")
async def bagofwords(request: TextRequest):
    '''Преобразование текста с помощью "мешка слов"'''
    return handle_text_with_bag_of_words(request)


@app.post("/tfidf")
async def tfidf(request: TextRequest):
    '''Преобразование текста с помощью TF-IDF'''
    return handle_text_with_tfidf(request)

@app.post("/lsa")
async def lsa(request: TextRequest):
    '''Латентный семантический анализ (LSA)'''
    return handle_text_with_lsa(request)


@app.post("/word2vec")
async def word2vec(request: TextRequest):
    '''Упрощённое представление Word2Vec'''
    return handle_text_with_word2vec(request)

@app.post("/textnltk/tokenize")
async def tokenize_text_endpoint(request: TextRequest):
    '''Токенизация текста'''
    request_data = request.model_dump()
    return process_text_request(request_data, "tokenize")

@app.post("/textnltk/stem")
async def stem_text_endpoint(request: TextRequest, stemmer_type: str = "porter"):
    '''Стемминг текста'''
    request_data = request.model_dump()
    request_data["stemmer_type"] = stemmer_type
    return process_text_request(request_data, "stem")

@app.post("/textnltk/lemmatize")
async def lemmatize_text_endpoint(request: TextRequest, pos_tagging: bool = True):
    '''Лемматизация текста'''
    request_data = request.model_dump()
    request_data["pos_tagging"] = pos_tagging
    return process_text_request(request_data, "lemmatize")

@app.post("/textnltk/postag")
async def postag_text_endpoint(request: TextRequest, detailed: bool = False):
    '''POS-теггинг текста'''
    request_data = request.model_dump()
    request_data["detailed"] = detailed
    return process_text_request(request_data, "postag")

@app.post("/textnltk/ner")
async def ner_text_endpoint(request: TextRequest, binary: bool = False):
    '''Распознавание именованных сущностей'''
    request_data = request.model_dump()
    request_data["binary"] = binary
    return process_text_request(request_data, "ner")