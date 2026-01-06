from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    '''Корневой эндпоинт'''
    return {"message": "Hello, FastAPI !!!"}
