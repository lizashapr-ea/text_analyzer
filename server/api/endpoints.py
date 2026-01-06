from fastapi import FastAPI
from server.models.models import TextRequest
from server.preprocessing.bagofwords import handle_text_with_bag_of_words

app = FastAPI()

@app.get("/")
async def root():
    '''Корневой эндпоинт'''
    return {"message": "Hello, FastAPI !!!"}

@app.get("/bagofwords")
async def bagofwords(request: TextRequest):
    '''Преобразование текста с помощь "мешка слов"'''
    return handle_text_with_bag_of_words(request)