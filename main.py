from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from typing import List
app = FastAPI()

class TextRequest(BaseModel):
    documents: List[str]

@app.get("/")
async def root():
    '''Корневой эндпоинт'''
    return {"message": "Hello, FastAPI !!!"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # экземпляр нашего приложения
        host="127.0.0.1",  # слушать на всех интерфейсах
        port=8001,      # порт для подключения
        reload=True     # автоматическая перезагрузка при изменениях
    )