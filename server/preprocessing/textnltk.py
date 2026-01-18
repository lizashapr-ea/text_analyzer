# TODO реализовать функции: токенизации, стемминга, лемматизации, Part of Speech (POS) tagging, Named Entity Recognition.
# Важно! Каждая функция пойдет под своим endpoint'ом в соответствующем файле (endpoints.py) с началом в виде /textnltk/...

from server.models.models import TextRequest

import nltk
import string
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import ssl
from pymystem3 import Mystem
import re
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc

# SSL fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка только необходимых ресурсов для русского
def download_nltk_resources():
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Не удалось загрузить '{resource}': {e}")
    # Загружаем русские стоп-слова
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Не удалось загрузить stopwords: {e}")
    print("Ресурсы NLTK для русского загружены")

download_nltk_resources()

# Инструменты (только для русского)
mystem = Mystem()
snowball_stemmer_ru = SnowballStemmer("russian")
stop_words_ru = set(stopwords.words('russian'))

# Глобальные компоненты Natasha
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Язык фиксирован
def detect_language(text: str) -> str:
    return 'ru'

def tokenize_text(text: str, remove_punct: bool = True, remove_stopwords: bool = False) -> Dict[str, Any]:
    try:
        lang = 'ru'
        stop_words = stop_words_ru
        
        sentences = sent_tokenize(text, language='russian')  # ← явно указываем язык!
        all_tokens = []
        sentence_tokens = []

        for sentence in sentences:
            tokens = word_tokenize(sentence, language='russian')
            cleaned_tokens = []
            for token in tokens:
                if remove_punct and token in string.punctuation:
                    continue
                if remove_stopwords and token.lower() in stop_words:
                    continue
                cleaned_tokens.append(token)
                all_tokens.append(token)
            
            sentence_tokens.append({
                "sentence": sentence,
                "tokens": cleaned_tokens,
                "token_count": len(cleaned_tokens)
            })
        
        total_words = len(all_tokens)
        unique_words = len(set(t.lower() for t in all_tokens))
        
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
        return {"success": False, "error": str(e)}
    

def stem_text(text: str, remove_punct: bool = True) -> Dict[str, Any]:
    """
    Стемминг текста (только русский язык)
    
    Args:
        text: Входной текст на русском языке
        remove_punct: Удалить пунктуацию
    
    Returns:
        Словарь с результатами стемминга
    """
    try:
        lang = 'ru'
        stemmer = snowball_stemmer_ru
        stemmer_used = "snowball_russian"
        
        # Токенизация (указываем язык явно)
        tokens = word_tokenize(text, language='russian')
        
        # Стемминг
        original_stemmed_pairs = []
        
        for token in tokens:
            if remove_punct and token in string.punctuation:
                continue
            stemmed = stemmer.stem(token)
            original_stemmed_pairs.append({
                "original": token,
                "stemmed": stemmed
            })
        
        if not original_stemmed_pairs:
            return {
                "success": True,
                "language": lang,
                "stemmer_type": stemmer_used,
                "total_tokens": len(tokens),
                "stemmed_tokens": 0,
                "unique_stems": 0,
                "reduction_rate": 0.0,
                "original_stemmed_pairs": [],
                "stem_groups": {},
                "stemmed_text": ""
            }
        
        stemmed_tokens = [pair["stemmed"] for pair in original_stemmed_pairs]
        
        # Группировка по основам
        stem_groups = {}
        for pair in original_stemmed_pairs:
            stem = pair["stemmed"]
            if stem not in stem_groups:
                stem_groups[stem] = []
            stem_groups[stem].append(pair["original"])
        
        # Коэффициент сжатия
        original_unique = len(set(pair["original"] for pair in original_stemmed_pairs))
        stemmed_unique = len(set(stemmed_tokens))
        reduction_rate = 1 - (stemmed_unique / original_unique) if original_unique > 0 else 0
        
        return {
            "success": True,
            "language": lang,
            "stemmer_type": stemmer_used,
            "total_tokens": len(tokens),
            "stemmed_tokens": len(stemmed_tokens),
            "unique_stems": stemmed_unique,
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
    Лемматизация текста (только русский язык через Mystem)
    
    Args:
        text: Входной текст на русском языке
        pos_tagging: Игнорируется (сохранён для совместимости API)
        remove_punct: Удалить пунктуацию
    
    Returns:
        Словарь с результатами лемматизации
    """
    try:
        lang = 'ru'
        method_used = "mystem"
        
        # Анализ через Mystem
        analysis = mystem.analyze(text)
        
        lemmatized_tokens = []
        lemmatization_pairs = []
        
        for item in analysis:
            original = item.get('text', '')
            if not original.strip():
                continue
            
            # Пропускаем пунктуацию
            if remove_punct and original in string.punctuation:
                continue
            
            lemma = original  # по умолчанию
            
            # Если есть морфологический анализ — берём первую лемму
            if 'analysis' in item and item['analysis']:
                lemma = item['analysis'][0]['lex']
            
            lemmatized_tokens.append(lemma)
            lemmatization_pairs.append({
                "original": original,
                "lemma": lemma,
                "method": "mystem"
            })
        
        # Группировка по леммам
        lemma_groups = {}
        for pair in lemmatization_pairs:
            lemma = pair["lemma"]
            if lemma not in lemma_groups:
                lemma_groups[lemma] = []
            lemma_groups[lemma].append({"original": pair["original"]})
        
        total_tokens = len(lemmatization_pairs)
        unique_lemmas = len(set(lemmatized_tokens))
        
        return {
            "success": True,
            "language": lang,
            "method_used": method_used,
            "pos_tagging_used": False,  # Mystem не возвращает POS в этом формате
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
    POS-теггинг текста (только русский язык через Mystem)
    
    Args:
        text: Входной текст на русском языке
        detailed: Возвращать подробную информацию о тегах
    
    Returns:
        Словарь с результатами POS-теггинга
    """
    try:
        lang = 'ru'
        method_used = "mystem"
        
        # Анализ через Mystem
        analysis = mystem.analyze(text)
        
        pos_tags = []
        tag_counts = {}
        tag_examples = {}
        
        for item in analysis:
            original = item.get('text', '')
            if not original.strip():
                continue
            
            # Определяем тег по грамматической информации из Mystem
            gram_info = item['analysis'][0].get('gr', '')

            # Определяем часть речи по началу граммемы (до запятой или '=')
            if gram_info.startswith('S,') or gram_info.startswith('S='):
                tag = 'NOUN'
            elif gram_info.startswith('A,') or gram_info.startswith('A='):
                tag = 'ADJ'
            elif gram_info.startswith('V,') or gram_info.startswith('V='):
                tag = 'VERB'
            elif 'ADV' in gram_info:
                tag = 'ADV'
            elif gram_info.startswith('PR,') or gram_info.startswith('PR='):
                tag = 'ADP'
            elif 'CONJ' in gram_info:
                tag = 'CONJ'
            elif 'NUM' in gram_info:
                tag = 'NUM'
            elif 'SPRO' in gram_info:
                tag = 'PRON'          # ← местоимения!
            elif 'PART' in gram_info:
                tag = 'PART'          # ← частицы ("не", "же", "ли" и т.д.)
            elif 'INTJ' in gram_info:
                tag = 'INTJ'          # междометия
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
            'PRON': 'местоимение',      
            'PART': 'частица',          
            'INTJ': 'междометие',       
            'PUNCT': 'знак препинания',
            'X': 'другое'
        }

        # Формируем список тегов
        tag_list = []
        for tag, count in tag_counts.items():
            tag_info = {
                "tag": tag,
                "count": count,
                "examples": tag_examples[tag][:5]
            }
            if detailed:
                tag_info["description"] = tag_descriptions_ru.get(tag, "Неизвестный тег")
            tag_list.append(tag_info)
        
        tag_list.sort(key=lambda x: x["count"], reverse=True)
        
        # Группировка по категориям
        category_mapping = {
            'NOUN': 'Существительные',
            'VERB': 'Глаголы',
            'ADJ': 'Прилагательные',
            'ADV': 'Наречия',
            'ADP': 'Предлоги',
            'CONJ': 'Союзы',
            'NUM': 'Числительные',
            'PRON': 'Местоимения',      # ←
            'PART': 'Частицы',          # ←
            'INTJ': 'Междометия',       # ←
            'PUNCT': 'Знаки препинания'
        }
        
        category_counts = {}
        for tag, count in tag_counts.items():
            category = category_mapping.get(tag, 'Другое')
            category_counts[category] = category_counts.get(category, 0) + count
        
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

def ner_text(text: str) -> Dict[str, Any]:
    """
    Распознавание именованных сущностей (Named Entity Recognition) — только русский язык
    
    Args:
        text: Входной текст на русском языке
    
    Returns:
        Словарь с результатами NER
    """
    try:
        lang = 'ru'
        
        # Токенизация для подсчёта total_tokens (можно также использовать len(text.split()), но оставим как есть)
        tokens = word_tokenize(text, language='russian')
        
        # Обработка через Natasha
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        entities = []
        for span in doc.spans:
            # Преобразуем типы Natasha в стандартные
            type_map = {
                'PER': 'PERSON',
                'ORG': 'ORGANIZATION',
                'LOC': 'LOCATION'
            }
            entity_type = type_map.get(span.type, span.type)

            # Извлекаем токены
            span_tokens = [token.text for token in span.tokens]

            entities.append({
                "entity": span.text,
                "type": entity_type,
                "tokens": span_tokens
            })

        method_used = "natasha"

        # Статистика по типам
        entity_types = {}
        for ent in entities:
            t = ent["type"]
            entity_types[t] = entity_types.get(t, 0) + 1

        # Описания типов 
        entity_type_descriptions = {
            'PERSON': 'Люди, включая вымышленных',
            'ORGANIZATION': 'Компании, агентства, учреждения',
            'LOCATION': 'Местоположения: города, страны, регионы, улицы и т.д.'
        }

        detailed_entity_types = []
        for etype, count in entity_types.items():
            examples = [e["entity"] for e in entities if e["type"] == etype][:3]
            detailed_entity_types.append({
                "type": etype,
                "count": count,
                "description": entity_type_descriptions.get(etype, "Неизвестный тип сущности"),
                "examples": examples
            })

        detailed_entity_types.sort(key=lambda x: x["count"], reverse=True)

        # Подсветка сущностей в тексте
        highlighted_text = text
        # Сортируем по длине (длинные — первыми), чтобы избежать частичных замен
        for ent in sorted(entities, key=lambda x: len(x["entity"]), reverse=True):
            if ent["entity"] in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    ent["entity"],
                    f"[{ent['entity']}]({ent['type']})"
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
    Обработка запроса на обработку текста (только русский язык)
    
    Args:
        request_data: Данные запроса
        endpoint: Тип обработки (tokenize, stem, lemmatize, postag, ner)
    
    Returns:
        Результаты обработки
    """
    try:
        documents = request_data.get("documents", [])
        
        if not documents:
            return {
                "success": False,
                "error": "No documents provided"
            }
        
        results = []
        
        for doc in documents:
            if not isinstance(doc, str):
                continue
            
            # Выбор функции обработки (только русский)
            if endpoint == "tokenize":
                remove_punct = request_data.get("remove_punct", True)
                remove_stopwords = request_data.get("remove_stopwords", False)
                result = tokenize_text(doc, remove_punct=remove_punct, remove_stopwords=remove_stopwords)
                
            elif endpoint == "stem":
                # Параметр stemmer_type больше не используется (только snowball_russian)
                remove_punct = request_data.get("remove_punct", True)
                result = stem_text(doc, remove_punct=remove_punct)
                
            elif endpoint == "lemmatize":
                # pos_tagging игнорируется (Mystem всегда использует морфологию)
                remove_punct = request_data.get("remove_punct", True)
                result = lemmatize_text(doc, remove_punct=remove_punct)
                
            elif endpoint == "postag":
                detailed = request_data.get("detailed", False)
                result = pos_tag_text(doc, detailed=detailed)
                
            elif endpoint == "ner":
                # binary игнорируется (Natasha не поддерживает бинарный режим как NLTK)
                result = ner_text(doc)
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown endpoint: {endpoint}"
                }
            
            results.append({
                "document": doc[:100] + "..." if len(doc) > 100 else doc,
                "result": result
            })
        
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