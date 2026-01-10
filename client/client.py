#TODO Написать программу на Python с методом main для запросов к серверу.

import json
from urllib.request import urlopen, Request

url = "http://127.0.0.1:8001/"

with urlopen(url) as response:
    body = response.read()          # bytes
    result = json.loads(body.decode())  # dict

print(result)


url1 = "http://127.0.0.1:8001/bagofwords"

# Пример текста для обработки
payload = {
    "documents": ["Привет! Это пример текста для преобразования с помощью мешка слов."]
}

# Подготовка запроса
data = json.dumps(payload).encode('utf-8')
req = Request(url1, data=data, headers={'Content-Type': 'application/json'})

# Отправка запроса и получение ответа
with urlopen(req) as response:
    body = response.read()  # bytes
    result = json.loads(body.decode('utf-8'))  # dict

print(result)