#TODO Написать программу на Python с методом main для запросов к серверу.

import json
from urllib.request import urlopen

url = "http://127.0.0.1:8001/"

with urlopen(url) as response:
    body = response.read()          # bytes
    data = json.loads(body.decode())  # dict

print(data)
