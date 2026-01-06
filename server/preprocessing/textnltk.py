#TODO реализовать функции: токенизации, стемминга, лемматизации, Part of Speech (POS) tagging, Named Entity Recognition.
#Важно! Каждая функция пойдет под своим endpoint'ом в соответствующем файле (endpoints.py) с началом в виде /textnltk/...
#Например, "@app.get("/textnltk/tokenize")"

from server.models.models import TextRequest