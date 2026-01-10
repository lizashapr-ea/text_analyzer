#TODO реализовать функции: токенизации, стемминга, лемматизации, Part of Speech (POS) tagging, Named Entity Recognition.
#Важно! Каждая функция пойдет под своим endpoint'ом в соответствующем файле (endpoints.py) с началом в виде /textnltk/...
#Например, "@app.get("/textnltk/tokenize")"

from server.models.models import TextRequest

import nltk
import string
from typing import List, Dict, Any, Tuple
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tree import Tree
import ssl
from pymystem3 import Mystem
import re

# Настройки для обхода SSL при загрузке данных NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка необходимых ресурсов NLTK
def download_nltk_resources():
    """Загрузка необходимых ресурсов NLTK"""
    resources = [
        'punkt',
        'punkt_tab',  # Для русского токенизатора
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'wordnet',
        'stopwords',
        'omw-eng',  # Open Multilingual Wordnet
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Не удалось загрузить ресурс '{resource}': {e}")
    
    # Для русского языка загружаем отдельно
    try:
        # Пытаемся загрузить русские стоп-слова
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    print("Ресурсы NLTK загружены")

# Вызов загрузки ресурсов при импорте
download_nltk_resources()

# Инициализация инструментов для обоих языков
porter_stemmer = PorterStemmer()
snowball_stemmer_en = SnowballStemmer("english")
snowball_stemmer_ru = SnowballStemmer("russian")
mystem = Mystem()  # Для русского

# Стоп-слова для обоих языков
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))

# Функция для определения языка
def detect_language(text: str) -> str:
    """Простое определение языка текста"""
    # Простая эвристика для определения языка
    ru_chars = len([c for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я'])
    en_chars = len([c for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z'])
    
    if ru_chars > en_chars:
        return 'ru'
    else:
        return 'en'

def tokenize_text(text: str, remove_punct: bool = True, remove_stopwords: bool = False) -> Dict[str, Any]:
    """
    Токенизация текста
    
    Args:
        text: Входной текст
        remove_punct: Удалить пунктуацию
        remove_stopwords: Удалить стоп-слова
    
    Returns:
        Словарь с результатами токенизации
    """
    try:
        # Определяем язык
        lang = detect_language(text)
        
        # Выбираем стоп-слова в зависимости от языка
        stop_words = stop_words_ru if lang == 'ru' else stop_words_en
        
        # Токенизация по предложениям
        sentences = sent_tokenize(text)
        
        # Токенизация по словам
        all_tokens = []
        sentence_tokens = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Очистка токенов
            cleaned_tokens = []
            for token in tokens:
                # Удаление пунктуации
                if remove_punct and token in string.punctuation:
                    continue
                
                # Удаление стоп-слов
                if remove_stopwords and token.lower() in stop_words:
                    continue
                
                cleaned_tokens.append(token)
                all_tokens.append(token)
            
            sentence_tokens.append({
                "sentence": sentence,
                "tokens": cleaned_tokens,
                "token_count": len(cleaned_tokens)
            })
        
        # Статистика
        total_words = len(all_tokens)
        unique_words = len(set([token.lower() for token in all_tokens]))
        
        return {
            "success": True,
            "language": lang,
            "total_sentences": len(sentences),
            "total_words": total_words,
            "unique_words": unique_words,
            "sentences": sentence_tokens,
            "all_tokens": all_tokens
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def stem_text(text: str, stemmer_type: str = "porter", remove_punct: bool = True) -> Dict[str, Any]:
    """
    Стемминг текста
    
    Args:
        text: Входной текст
        stemmer_type: Тип стеммера ("porter" или "snowball")
        remove_punct: Удалить пунктуацию
    
    Returns:
        Словарь с результатами стемминга
    """
    try:
        # Определяем язык
        lang = detect_language(text)
        
        # Выбираем стеммер в зависимости от языка
        if lang == 'ru':
            if stemmer_type.lower() == "porter":
                # Porter не поддерживает русский, используем Snowball
                stemmer = snowball_stemmer_ru
                stemmer_used = "snowball_russian"
            else:
                stemmer = snowball_stemmer_ru
                stemmer_used = "snowball_russian"
        else:
            if stemmer_type.lower() == "snowball":
                stemmer = snowball_stemmer_en
                stemmer_used = "snowball_english"
            else:
                stemmer = porter_stemmer
                stemmer_used = "porter"
        
        # Токенизация
        tokens = word_tokenize(text)
        
        # Стемминг
        stemmed_tokens = []
        original_stemmed_pairs = []
        
        for token in tokens:
            # Пропускаем пунктуацию
            if remove_punct and token in string.punctuation:
                continue
            
            # Стемминг
            stemmed = stemmer.stem(token)
            stemmed_tokens.append(stemmed)
            original_stemmed_pairs.append({
                "original": token,
                "stemmed": stemmed
            })
        
        # Группировка одинаковых основ
        stem_groups = {}
        for pair in original_stemmed_pairs:
            stem = pair["stemmed"]
            if stem not in stem_groups:
                stem_groups[stem] = []
            stem_groups[stem].append(pair["original"])
        
        # Подсчет статистики
        reduction_rate = 1 - (len(set(stemmed_tokens)) / len(set([p["original"] for p in original_stemmed_pairs]))) if original_stemmed_pairs else 0
        
        return {
            "success": True,
            "language": lang,
            "stemmer_type": stemmer_used,
            "total_tokens": len(tokens),
            "stemmed_tokens": len(stemmed_tokens),
            "unique_stems": len(set(stemmed_tokens)),
            "reduction_rate": round(reduction_rate, 4),
            "original_stemmed_pairs": original_stemmed_pairs,
            "stem_groups": stem_groups,
            "stemmed_text": " ".join(stemmed_tokens)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def lemmatize_text(text: str, pos_tagging: bool = True, remove_punct: bool = True) -> Dict[str, Any]:
    """
    Лемматизация текста
    
    Args:
        text: Входной текст
        pos_tagging: Использовать POS-тегинг для лучшей лемматизации
        remove_punct: Удалить пунктуацию
    
    Returns:
        Словарь с результатами лемматизации
    """
    try:
        # Определяем язык
        lang = detect_language(text)
        
        if lang == 'ru':
            # Для русского используем Mystem
            # Mystem возвращает леммы с дополнительной информацией
            analysis = mystem.analyze(text)
            
            # Извлекаем леммы
            lemmatized_tokens = []
            lemmatization_pairs = []
            
            for item in analysis:
                if 'analysis' in item and item['analysis']:
                    # Берем первую гипотезу анализа
                    lemma = item['analysis'][0]['lex']
                    original = item['text']
                    
                    # Пропускаем пунктуацию
                    if remove_punct and original in string.punctuation:
                        continue
                    
                    lemmatized_tokens.append(lemma)
                    lemmatization_pairs.append({
                        "original": original,
                        "lemma": lemma,
                        "method": "mystem"
                    })
                elif 'text' in item and item['text'].strip():
                    # Если анализа нет, используем оригинальный текст
                    original = item['text']
                    if remove_punct and original in string.punctuation:
                        continue
                    
                    lemmatized_tokens.append(original)
                    lemmatization_pairs.append({
                        "original": original,
                        "lemma": original,
                        "method": "mystem"
                    })
            
            method_used = "mystem"
            
        else:
            # Для английского используем WordNetLemmatizer
            tokens = word_tokenize(text)
            
            if pos_tagging:
                # Получение POS-тегов
                pos_tags = pos_tag(tokens)
                
                # Функция для преобразования POS-тегов NLTK в формат WordNet
                def get_wordnet_pos(treebank_tag: str) -> str:
                    if treebank_tag.startswith('J'):
                        return 'a'  # adjective
                    elif treebank_tag.startswith('V'):
                        return 'v'  # verb
                    elif treebank_tag.startswith('N'):
                        return 'n'  # noun
                    elif treebank_tag.startswith('R'):
                        return 'r'  # adverb
                    else:
                        return 'n'  # по умолчанию noun
                
                # Лемматизация с учетом POS-тегов
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = []
                lemmatization_pairs = []
                
                for token, tag in pos_tags:
                    # Пропускаем пунктуацию
                    if remove_punct and token in string.punctuation:
                        continue
                    
                    # Получение POS-тега WordNet
                    wordnet_tag = get_wordnet_pos(tag)
                    
                    # Лемматизация
                    lemma = lemmatizer.lemmatize(token, pos=wordnet_tag)
                    lemmatized_tokens.append(lemma)
                    lemmatization_pairs.append({
                        "original": token,
                        "pos_tag": tag,
                        "wordnet_pos": wordnet_tag,
                        "lemma": lemma,
                        "method": "wordnet"
                    })
            else:
                # Лемматизация без POS-тегов
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = []
                lemmatization_pairs = []
                
                for token in tokens:
                    # Пропускаем пунктуацию
                    if remove_punct and token in string.punctuation:
                        continue
                    
                    # Лемматизация
                    lemma = lemmatizer.lemmatize(token)
                    lemmatized_tokens.append(lemma)
                    lemmatization_pairs.append({
                        "original": token,
                        "lemma": lemma,
                        "method": "wordnet"
                    })
            
            method_used = "wordnet" + ("_with_postag" if pos_tagging else "")
        
        # Группировка лемм
        lemma_groups = {}
        for pair in lemmatization_pairs:
            lemma = pair["lemma"]
            if lemma not in lemma_groups:
                lemma_groups[lemma] = []
            
            group_entry = {"original": pair["original"]}
            if "pos_tag" in pair:
                group_entry["pos_tag"] = pair["pos_tag"]
            
            lemma_groups[lemma].append(group_entry)
        
        # Подсчет статистики
        total_tokens = len(lemmatization_pairs)
        unique_lemmas = len(set(lemmatized_tokens))
        
        return {
            "success": True,
            "language": lang,
            "method_used": method_used,
            "pos_tagging_used": pos_tagging if lang == 'en' else False,
            "total_tokens": total_tokens,
            "lemmatized_tokens": len(lemmatized_tokens),
            "unique_lemmas": unique_lemmas,
            "lemmatization_pairs": lemmatization_pairs,
            "lemma_groups": lemma_groups,
            "lemmatized_text": " ".join(lemmatized_tokens)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def pos_tag_text(text: str, detailed: bool = False) -> Dict[str, Any]:
    """
    POS-теггинг текста
    
    Args:
        text: Входной текст
        detailed: Возвращать подробную информацию о тегах
    
    Returns:
        Словарь с результатами POS-теггинга
    """
    try:
        # Определяем язык
        lang = detect_language(text)
        
        if lang == 'ru':
            # Для русского используем Mystem для получения POS-тегов
            analysis = mystem.analyze(text)
            
            pos_tags = []
            tag_counts = {}
            tag_examples = {}
            
            for item in analysis:
                if 'analysis' in item and item['analysis']:
                    original = item['text']
                    # Mystem возвращает грамматическую информацию
                    gram_info = item['analysis'][0].get('gr', '')
                    
                    # Извлекаем часть речи из грамматической информации
                    if 'S' in gram_info:  # Существительное
                        tag = 'NOUN'
                    elif 'V' in gram_info:  # Глагол
                        tag = 'VERB'
                    elif 'A' in gram_info:  # Прилагательное
                        tag = 'ADJ'
                    elif 'ADV' in gram_info:  # Наречие
                        tag = 'ADV'
                    elif 'PR' in gram_info:  # Предлог
                        tag = 'ADP'
                    elif 'CONJ' in gram_info:  # Союз
                        tag = 'CONJ'
                    elif 'NUM' in gram_info:  # Числительное
                        tag = 'NUM'
                    else:
                        tag = 'X'  # Неизвестно
                else:
                    original = item['text']
                    if original in string.punctuation:
                        tag = 'PUNCT'
                    else:
                        tag = 'X'
                
                pos_tags.append((original, tag))
                
                # Статистика
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                    tag_examples[tag] = []
                
                tag_counts[tag] += 1
                if original not in tag_examples[tag]:
                    tag_examples[tag].append(original)
            
            # Описания тегов для русского
            tag_descriptions_ru = {
                'NOUN': 'существительное',
                'VERB': 'глагол',
                'ADJ': 'прилагательное',
                'ADV': 'наречие',
                'ADP': 'предлог',
                'CONJ': 'союз',
                'NUM': 'числительное',
                'PUNCT': 'знак препинания',
                'X': 'другое'
            }
            
            method_used = "mystem"
            
        else:
            # Для английского используем NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Статистика по тегам
            tag_counts = {}
            tag_examples = {}
            
            for token, tag in pos_tags:
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                    tag_examples[tag] = []
                
                tag_counts[tag] += 1
                if token not in tag_examples[tag]:
                    tag_examples[tag].append(token)
            
            # Описания тегов для английского
            tag_descriptions_en = {
                'NN': 'noun, singular or mass',
                'NNS': 'noun, plural',
                'NNP': 'proper noun, singular',
                'NNPS': 'proper noun, plural',
                'VB': 'verb, base form',
                'VBD': 'verb, past tense',
                'VBG': 'verb, gerund or present participle',
                'VBN': 'verb, past participle',
                'VBP': 'verb, non-3rd person singular present',
                'VBZ': 'verb, 3rd person singular present',
                'JJ': 'adjective',
                'JJR': 'adjective, comparative',
                'JJS': 'adjective, superlative',
                'RB': 'adverb',
                'RBR': 'adverb, comparative',
                'RBS': 'adverb, superlative',
                'IN': 'preposition or subordinating conjunction',
                'DT': 'determiner',
                'PRP': 'personal pronoun',
                'PRP$': 'possessive pronoun',
                'CC': 'coordinating conjunction',
                'CD': 'cardinal number',
                'MD': 'modal',
                'TO': 'to',
                'UH': 'interjection',
                'WP': 'wh-pronoun',
                'WRB': 'wh-adverb',
                '.': 'punctuation mark, sentence closer',
                ',': 'punctuation mark, comma',
                ':': 'punctuation mark, colon',
                ';': 'punctuation mark, semicolon'
            }
            
            method_used = "nltk"
        
        # Список POS-тегов с примерами
        tag_list = []
        for tag, count in tag_counts.items():
            tag_info = {
                "tag": tag,
                "count": count,
                "examples": tag_examples[tag][:5]  # Первые 5 примеров
            }
            
            if detailed:
                # Добавление описания тега
                if lang == 'ru':
                    tag_info["description"] = tag_descriptions_ru.get(tag, "Неизвестный тег")
                else:
                    tag_info["description"] = tag_descriptions_en.get(tag, "Unknown tag")
            
            tag_list.append(tag_info)
        
        # Сортировка по количеству
        tag_list.sort(key=lambda x: x["count"], reverse=True)
        
        # Группировка по основным категориям (для английского)
        if lang == 'en':
            category_mapping = {
                'NN': 'Nouns', 'NNS': 'Nouns', 'NNP': 'Nouns', 'NNPS': 'Nouns',
                'VB': 'Verbs', 'VBD': 'Verbs', 'VBG': 'Verbs', 'VBN': 'Verbs', 'VBP': 'Verbs', 'VBZ': 'Verbs',
                'JJ': 'Adjectives', 'JJR': 'Adjectives', 'JJS': 'Adjectives',
                'RB': 'Adverbs', 'RBR': 'Adverbs', 'RBS': 'Adverbs'
            }
            
            category_counts = {}
            for tag, count in tag_counts.items():
                category = category_mapping.get(tag, 'Other')
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += count
        else:
            # Для русского
            category_mapping = {
                'NOUN': 'Существительные',
                'VERB': 'Глаголы',
                'ADJ': 'Прилагательные',
                'ADV': 'Наречия',
                'ADP': 'Предлоги',
                'CONJ': 'Союзы',
                'NUM': 'Числительные',
                'PUNCT': 'Знаки препинания'
            }
            
            category_counts = {}
            for tag, count in tag_counts.items():
                category = category_mapping.get(tag, 'Другое')
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += count
        
        return {
            "success": True,
            "language": lang,
            "method_used": method_used,
            "total_tokens": len(pos_tags),
            "pos_tags": pos_tags,
            "tag_statistics": tag_list,
            "category_counts": category_counts,
            "tagged_text": " ".join([f"{token}/{tag}" for token, tag in pos_tags])
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def ner_text(text: str, binary: bool = False) -> Dict[str, Any]:
    """
    Распознавание именованных сущностей (Named Entity Recognition)
    
    Args:
        text: Входной текст
        binary: Бинарный режим (только наличие сущностей без классификации)
    
    Returns:
        Словарь с результатами NER
    """
    try:
        # Определяем язык
        lang = detect_language(text)
        
        if lang == 'ru':
            # Для русского используем простое правило-основу
            # В реальном проекте нужно использовать natasha или другой NER для русского
            tokens = word_tokenize(text)
            
            # Простая эвристика для русского
            entities = []
            
            # Поиск возможных имен (слова с заглавной буквы в середине предложения)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                i = 0
                while i < len(words):
                    if words[i][0].isupper() and i > 0:  # Не первое слово в предложении
                        # Собираем последовательность слов с заглавной буквы
                        entity_tokens = [words[i]]
                        j = i + 1
                        while j < len(words) and words[j][0].isupper():
                            entity_tokens.append(words[j])
                            j += 1
                        
                        if len(entity_tokens) >= 1:
                            entity_name = " ".join(entity_tokens)
                            # Простая классификация
                            if any(word in entity_name.lower() for word in ['ул.', 'улица', 'проспект', 'пр.', 'город', 'г.', 'страна']):
                                entity_type = 'LOCATION'
                            elif any(word in entity_name.lower() for word in ['ооо', 'зао', 'оао', 'компания', 'корпорация']):
                                entity_type = 'ORGANIZATION'
                            else:
                                entity_type = 'PERSON' if len(entity_tokens) <= 3 else 'ORGANIZATION'
                            
                            entities.append({
                                "entity": entity_name,
                                "type": entity_type,
                                "tokens": entity_tokens
                            })
                        
                        i = j
                    else:
                        i += 1
            
            method_used = "heuristic_russian"
            
        else:
            # Для английского используем NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Распознавание именованных сущностей
            named_entities = ne_chunk(pos_tags, binary=binary)
            
            # Извлечение сущностей из дерева
            entities = []
            
            def extract_entities(tree):
                entities_found = []
                
                if hasattr(tree, 'label'):
                    if tree.label():
                        # Это именованная сущность
                        entity_name = " ".join([child[0] for child in tree])
                        entity_type = tree.label()
                        
                        entities_found.append({
                            "entity": entity_name,
                            "type": entity_type,
                            "tokens": [child[0] for child in tree]
                        })
                    
                    # Рекурсивный обход детей
                    for child in tree:
                        if isinstance(child, Tree):
                            entities_found.extend(extract_entities(child))
                
                return entities_found
            
            entities = extract_entities(named_entities)
            method_used = "nltk"
        
        # Статистика по типам сущностей
        entity_types = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        # Сокращенные описания типов сущностей
        entity_type_descriptions = {
            'PERSON': 'People, including fictional',
            'ORGANIZATION': 'Companies, agencies, institutions',
            'GPE': 'Countries, cities, states (Geo-Political Entity)',
            'LOCATION': 'Non-GPE locations, mountain ranges, bodies of water',
            'DATE': 'Absolute or relative dates or periods',
            'TIME': 'Times smaller than a day',
            'MONEY': 'Monetary values, including unit',
            'PERCENT': 'Percentage (including "%")',
            'FACILITY': 'Buildings, airports, highways, bridges, etc.',
            'PRODUCT': 'Objects, vehicles, foods, etc. (not services)'
        }
        
        # Подробная информация о типах сущностей
        detailed_entity_types = []
        for entity_type, count in entity_types.items():
            type_info = {
                "type": entity_type,
                "count": count,
                "description": entity_type_descriptions.get(entity_type, "Unknown entity type"),
                "examples": [e["entity"] for e in entities if e["type"] == entity_type][:3]  # 3 примера
            }
            detailed_entity_types.append(type_info)
        
        # Сортировка по количеству
        detailed_entity_types.sort(key=lambda x: x["count"], reverse=True)
        
        # Подсветка сущностей в тексте
        highlighted_text = text
        for entity in sorted(entities, key=lambda x: len(x["entity"]), reverse=True):
            if entity["entity"] in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    entity["entity"],
                    f"[{entity['entity']}]({entity['type']})"
                )
        
        return {
            "success": True,
            "language": lang,
            "method_used": method_used,
            "total_tokens": len(tokens),
            "entities_found": len(entities),
            "entities": entities,
            "entity_types": detailed_entity_types,
            "highlighted_text": highlighted_text
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def process_text_request(request_data: dict, endpoint: str) -> Dict[str, Any]:
    """
    Обработка запроса на обработку текста
    
    Args:
        request_data: Данные запроса
        endpoint: Тип обработки (tokenize, stem, lemmatize, postag, ner)
    
    Returns:
        Результаты обработки
    """
    try:
        # Извлечение документов из запроса
        documents = request_data.get("documents", [])
        
        if not documents:
            return {
                "success": False,
                "error": "No documents provided"
            }
        
        # Обработка каждого документа
        results = []
        
        for doc in documents:
            if not isinstance(doc, str):
                continue
            
            # Выбор функции обработки
            if endpoint == "tokenize":
                result = tokenize_text(doc)
            elif endpoint == "stem":
                stemmer_type = request_data.get("stemmer_type", "porter")
                result = stem_text(doc, stemmer_type=stemmer_type)
            elif endpoint == "lemmatize":
                pos_tagging = request_data.get("pos_tagging", True)
                result = lemmatize_text(doc, pos_tagging=pos_tagging)
            elif endpoint == "postag":
                detailed = request_data.get("detailed", False)
                result = pos_tag_text(doc, detailed=detailed)
            elif endpoint == "ner":
                binary = request_data.get("binary", False)
                result = ner_text(doc, binary=binary)
            else:
                return {
                    "success": False,
                    "error": f"Unknown endpoint: {endpoint}"
                }
            
            results.append({
                "document": doc[:100] + "..." if len(doc) > 100 else doc,
                "result": result
            })
        
        # Сводная статистика
        summary = {
            "total_documents": len(documents),
            "processed_documents": len(results),
            "endpoint": endpoint
        }
        
        return {
            "success": True,
            "summary": summary,
            "results": results
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Примеры использования функций
if __name__ == "__main__":
    # Тестовые тексты на обоих языках
    test_text_en = "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. John Smith works at Google in New York City and earns $150,000 per year."
    
    test_text_ru = "Обработка естественного языка (NLP) - это область искусственного интеллекта, которая занимается взаимодействием между компьютерами и человеческим языком. Иван Иванов работает в компании Яндекс в Москве и зарабатывает 150 000 рублей в год."
    
    print("Тестирование функций NLTK:")
    print("=" * 60)
    
    print("\n=== АНГЛИЙСКИЙ ТЕКСТ ===")
    # Токенизация
    print("\n1. Токенизация:")
    token_result = tokenize_text(test_text_en)
    print(f"Язык: {token_result.get('language', 'unknown')}")
    print(f"Предложений: {token_result.get('total_sentences', 0)}")
    print(f"Слов: {token_result.get('total_words', 0)}")
    print(f"Уникальных слов: {token_result.get('unique_words', 0)}")
    
    # Стемминг
    print("\n2. Стемминг (Porter):")
    stem_result = stem_text(test_text_en)
    print(f"Язык: {stem_result.get('language', 'unknown')}")
    print(f"Оригинальных токенов: {stem_result.get('total_tokens', 0)}")
    print(f"Уникальных основ: {stem_result.get('unique_stems', 0)}")
    
    # Лемматизация
    print("\n3. Лемматизация:")
    lemma_result = lemmatize_text(test_text_en, pos_tagging=True)
    print(f"Язык: {lemma_result.get('language', 'unknown')}")
    print(f"Метод: {lemma_result.get('method_used', 'unknown')}")
    print(f"Уникальных лемм: {lemma_result.get('unique_lemmas', 0)}")
    
    print("\n=== РУССКИЙ ТЕКСТ ===")
    # Токенизация
    print("\n1. Токенизация:")
    token_result_ru = tokenize_text(test_text_ru)
    print(f"Язык: {token_result_ru.get('language', 'unknown')}")
    print(f"Предложений: {token_result_ru.get('total_sentences', 0)}")
    print(f"Слов: {token_result_ru.get('total_words', 0)}")
    print(f"Уникальных слов: {token_result_ru.get('unique_words', 0)}")
    
    # Стемминг
    print("\n2. Стемминг:")
    stem_result_ru = stem_text(test_text_ru)
    print(f"Язык: {stem_result_ru.get('language', 'unknown')}")
    print(f"Стеммер: {stem_result_ru.get('stemmer_type', 'unknown')}")
    print(f"Оригинальных токенов: {stem_result_ru.get('total_tokens', 0)}")
    
    # Лемматизация
    print("\n3. Лемматизация:")
    lemma_result_ru = lemmatize_text(test_text_ru)
    print(f"Язык: {lemma_result_ru.get('language', 'unknown')}")
    print(f"Метод: {lemma_result_ru.get('method_used', 'unknown')}")
    print(f"Уникальных лемм: {lemma_result_ru.get('unique_lemmas', 0)}")