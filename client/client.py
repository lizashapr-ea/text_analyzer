import json
from urllib.request import Request, urlopen
import os


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
    output_lines = []  # Сюда будем собирать весь вывод

    # --- читаем данные из файла ---
    file_path = os.path.join(os.path.dirname(__file__), "../server/ressources/corpus.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.loads(f.read())

    def log_and_collect(title, data):
        """Печатает заголовок и данные, и добавляет их в список для сохранения"""
        header = f"{title}:\n"
        formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
        full_output = header + formatted_data + "\n\n"
        print(header.strip())
        print(formatted_data)
        output_lines.append(full_output)

    # Выполняем запросы и собираем результаты
    log_and_collect("TOKENIZE", send_request(f"{base_url}/textnltk/tokenize", payload))
    log_and_collect("STEM", send_request(f"{base_url}/textnltk/stem?stemmer_type=porter", payload))
    log_and_collect("LEMMATIZE", send_request(f"{base_url}/textnltk/lemmatize?pos_tagging=true", payload))
    log_and_collect("POSTAG", send_request(f"{base_url}/textnltk/postag?detailed=false", payload))
    log_and_collect("NER", send_request(f"{base_url}/textnltk/ner?binary=false", payload))
    log_and_collect("BAG OF WORDS", send_request(f"{base_url}/bagofwords", payload))
    log_and_collect("TF-IDF", send_request(f"{base_url}/tfidf", payload))
    log_and_collect("LSA", send_request(f"{base_url}/lsa", payload))
    log_and_collect("WORD2VEC", send_request(f"{base_url}/word2vec", payload))

    # Сохраняем всё в файл
    output_file = os.path.join(os.path.dirname(__file__), "output.txt")
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("".join(output_lines))

    print(f"\nРезультаты сохранены в файл: {output_file}")


if __name__ == "__main__":
    main()
