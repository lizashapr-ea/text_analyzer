from fastapi import FastAPI
from server.models.models import TextRequest
from server.preprocessing.bagofwords import handle_text_with_bag_of_words
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


