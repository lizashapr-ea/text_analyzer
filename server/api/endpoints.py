from fastapi import FastAPI
from server.models.models import TextRequest
from server.preprocessing.bagofwords import handle_text_with_bag_of_words
from server.preprocessing.textnltk import process_text_request

app = FastAPI()

@app.get("/")
async def root():
    '''Корневой эндпоинт'''
    return {"message": "Hello, FastAPI !!!"}

@app.get("/bagofwords")
async def bagofwords(request: TextRequest):
    '''Преобразование текста с помощь "мешка слов"'''
    return handle_text_with_bag_of_words(request)

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

