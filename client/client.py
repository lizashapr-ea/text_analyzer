import json
from urllib.request import Request, urlopen


def send_request(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    with urlopen(req) as response:
        body = response.read()
        result = json.loads(body.decode("utf-8"))

    return result


def pretty_print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    base_url = "http://127.0.0.1:8001"

    payload = {
        "documents": [
            "Привет! Это пример текста для обработки с помощью клиент-серверной программы.", 
            "Я Лиза, мне шестнадцать лет, я люблю читать, аааааааа"
        ]
    }
    
    # tokenize
    url_tokenize = f"{base_url}/textnltk/tokenize"
    print("TOKENIZE:")
    pretty_print(send_request(url_tokenize, payload))
    print()

    # stem
    url_stem = f"{base_url}/textnltk/stem?stemmer_type=porter"
    print("STEM:")
    pretty_print(send_request(url_stem, payload))
    print()

    # lemmatize
    url_lemmatize = f"{base_url}/textnltk/lemmatize?pos_tagging=true"
    print("LEMMATIZE:")
    pretty_print(send_request(url_lemmatize, payload))
    print()

    # postag
    url_postag = f"{base_url}/textnltk/postag?detailed=false"
    print("POSTAG:")
    pretty_print(send_request(url_postag, payload))
    print()

    # ner
    url_ner = f"{base_url}/textnltk/ner?binary=false"
    print("NER:")
    pretty_print(send_request(url_ner, payload))
    print()

    # bagofwords
    url_bow = f"{base_url}/bagofwords"
    print("BAG OF WORDS:")
    pretty_print(send_request(url_bow, payload))
    print()

    # tfidf
    url_tfidf = f"{base_url}/tfidf"
    print("TF-IDF:")
    pretty_print(send_request(url_tfidf, payload))
    print()

    # lsa
    url_lsa = f"{base_url}/lsa"
    print("LSA:")
    pretty_print(send_request(url_lsa, payload))
    print()

    # word2vec
    url_word2vec = f"{base_url}/word2vec"
    print("WORD2VEC:")
    pretty_print(send_request(url_word2vec, payload))
    print()



if __name__ == "__main__":
    main()