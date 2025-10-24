#!/usr/bin/env python3
"""
Скрипт для подготовки датасета shridhar_maharaj_books для загрузки в Hugging Face Hub
"""

import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
import re

def clean_text(text):
    """Очистка текста от лишних пробелов и символов"""
    # Удаляем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы в начале и конце
    text = text.strip()
    return text

def extract_book_info(filename, content):
    """Извлечение информации о книге из содержимого"""
    lines = content.split('\n')
    
    # Ищем заголовок книги (обычно в первых 20 строках)
    title = "Неизвестная книга"
    author = "Шрила Бхакти Ракшак Шридхар Дев-Госвами Махарадж"
    
    for i, line in enumerate(lines[:20]):
        line = line.strip()
        if len(line) > 10 and len(line) < 100 and not line.startswith(' '):
            # Пропускаем служебные строки
            if not any(skip in line.lower() for skip in ['содержание', 'глава', 'от издателей', 'предисловие']):
                title = line
                break
    
    return {
        'filename': filename,
        'title': title,
        'author': author,
        'language': 'ru',
        'topic': 'spirituality',
        'religion': 'vaishnavism',
        'text_length': len(content),
        'word_count': len(content.split())
    }

def load_books(dataset_path):
    """Загрузка всех книг из датасета"""
    books = []
    
    # Получаем все txt файлы
    txt_files = list(Path(dataset_path).glob("*.txt"))
    
    for txt_file in txt_files:
        print(f"Обрабатываю файл: {txt_file.name}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Очищаем текст
            cleaned_content = clean_text(content)
            
            # Извлекаем информацию о книге
            book_info = extract_book_info(txt_file.name, content)
            
            # Создаем запись для датасета
            book_record = {
                'id': txt_file.stem,
                'text': cleaned_content,
                **book_info
            }
            
            books.append(book_record)
            
        except Exception as e:
            print(f"Ошибка при обработке файла {txt_file.name}: {e}")
            continue
    
    return books

def create_dataset(books):
    """Создание датасета в формате Hugging Face"""
    
    # Создаем основной датасет
    dataset = Dataset.from_list(books)
    
    # Разделяем на train/validation/test (80/10/10)
    train_size = int(0.8 * len(books))
    val_size = int(0.1 * len(books))
    
    train_books = books[:train_size]
    val_books = books[train_size:train_size + val_size]
    test_books = books[train_size + val_size:]
    
    # Создаем DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_books),
        'validation': Dataset.from_list(val_books),
        'test': Dataset.from_list(test_books)
    })
    
    return dataset_dict

def create_dataset_card():
    """Создание README.md для датасета"""
    card_content = """---
license: mit
language:
- ru
tags:
- spirituality
- vaishnavism
- krishna
- hinduism
- religious-texts
- russian
size_categories:
- 1K<n<10K
---

# Shridhar Maharaj Books Dataset

## Описание

Этот датасет содержит коллекцию духовных книг Шрилы Бхакти Ракшака Шридхара Дев-Госвами Махараджа на русском языке. Книги посвящены философии вайшнавизма, учению о Кришне и духовной практике.

## Содержимое

Датасет включает в себя:
- 15 текстовых файлов с книгами и лекциями
- Общий объем текста: ~3.5MB
- Язык: русский
- Тематика: духовность, вайшнавизм, кришнаизм

## Структура данных

Каждая запись содержит:
- `id`: уникальный идентификатор книги
- `text`: полный текст книги
- `title`: название книги
- `author`: автор (Шрила Бхакти Ракшак Шридхар Дев-Госвами Махарадж)
- `language`: язык (ru)
- `topic`: тема (spirituality)
- `religion`: религия (vaishnavism)
- `text_length`: длина текста в символах
- `word_count`: количество слов

## Использование

```python
from datasets import load_dataset

dataset = load_dataset("your-username/shridhar_maharaj_books")
```

## Авторские права

Все тексты принадлежат Шри Чайтанья Сарасват Матху и распространяются в соответствии с их лицензией.

## Контакт

Для вопросов по датасету обращайтесь к создателю репозитория.
"""
    
    return card_content

def main():
    """Основная функция"""
    dataset_path = "/Users/anton/proj/ai.nativemind.net/sridhar/datasets/shridhar_maharaj_books"
    
    print("Загружаю книги...")
    books = load_books(dataset_path)
    print(f"Загружено {len(books)} книг")
    
    print("Создаю датасет...")
    dataset_dict = create_dataset(books)
    
    print("Сохраняю датасет...")
    output_path = "/Users/anton/proj/ai.nativemind.net/sridhar/shridhar_maharaj_books_dataset"
    dataset_dict.save_to_disk(output_path)
    
    print("Создаю dataset card...")
    card_content = create_dataset_card()
    with open(f"{output_path}/README.md", 'w', encoding='utf-8') as f:
        f.write(card_content)
    
    print("Создаю метаданные...")
    metadata = {
        "dataset_name": "shridhar_maharaj_books",
        "description": "Коллекция духовных книг Шрилы Шридхара Махараджа на русском языке",
        "language": "ru",
        "total_books": len(books),
        "total_text_length": sum(book['text_length'] for book in books),
        "total_words": sum(book['word_count'] for book in books)
    }
    
    with open(f"{output_path}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Датасет сохранен в: {output_path}")
    print(f"Общее количество книг: {len(books)}")
    print(f"Общая длина текста: {metadata['total_text_length']:,} символов")
    print(f"Общее количество слов: {metadata['total_words']:,}")

if __name__ == "__main__":
    main()
