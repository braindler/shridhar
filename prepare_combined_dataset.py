#!/usr/bin/env python3
"""
Скрипт для подготовки объединенного датасета для обучения модели Шридхара Махараджа
Объединяет shridhar_maharaj_books и mozgach_alice_gift_sql_dataset
"""

import os
import json
import sqlite3
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и символов"""
    # Удаляем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы в начале и конце
    text = text.strip()
    return text

def load_shridhar_books(dataset_path: str) -> List[Dict[str, Any]]:
    """Загрузка книг Шридхара Махараджа"""
    books = []
    txt_files = list(Path(dataset_path).glob("*.txt"))
    
    for txt_file in txt_files:
        print(f"Загружаю книгу: {txt_file.name}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_content = clean_text(content)
            
            # Извлекаем информацию о книге
            lines = content.split('\n')
            title = "Неизвестная книга"
            
            for i, line in enumerate(lines[:20]):
                line = line.strip()
                if len(line) > 10 and len(line) < 100 and not line.startswith(' '):
                    if not any(skip in line.lower() for skip in ['содержание', 'глава', 'от издателей', 'предисловие']):
                        title = line
                        break
            
            book_record = {
                'id': f"shridhar_{txt_file.stem}",
                'text': cleaned_content,
                'source': 'shridhar_books',
                'title': title,
                'author': 'Шрила Бхакти Ракшак Шридхар Дев-Госвами Махарадж',
                'language': 'ru',
                'topic': 'spirituality',
                'religion': 'vaishnavism',
                'text_length': len(cleaned_content),
                'word_count': len(cleaned_content.split()),
                'context_type': 'spiritual_text'
            }
            
            books.append(book_record)
            
        except Exception as e:
            print(f"Ошибка при обработке файла {txt_file.name}: {e}")
            continue
    
    return books

def load_alice_sql_data(dataset_path: str) -> List[Dict[str, Any]]:
    """Загрузка данных из SQL дампа Alice"""
    dialogues = []
    
    # Создаем временную SQLite базу
    temp_db = "/tmp/alice_temp.db"
    
    try:
        # Объединяем все части SQL дампа
        print("Объединяю SQL дамп...")
        sql_files = sorted(Path(dataset_path).glob("alice_*"))
        
        with open("/tmp/alice_combined.sql", 'w', encoding='utf-8') as outfile:
            for sql_file in sql_files:
                print(f"Обрабатываю: {sql_file.name}")
                with open(sql_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    # Исправляем кодировку и формат
                    content = content.replace('utf8mb4', 'utf8')
                    content = content.replace('utf8mb3', 'utf8')
                    outfile.write(content)
        
        print("Импортирую в SQLite...")
        # Создаем SQLite соединение
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Читаем и выполняем SQL
        with open("/tmp/alice_combined.sql", 'r', encoding='utf-8') as f:
            sql_content = f.read()
            
        # Разбиваем на отдельные запросы
        sql_commands = sql_content.split(';')
        
        for i, command in enumerate(sql_commands):
            if command.strip():
                try:
                    cursor.execute(command)
                    if i % 100 == 0:
                        print(f"Выполнено команд: {i}")
                except Exception as e:
                    if "table" not in str(e).lower():
                        print(f"Ошибка в команде {i}: {e}")
                    continue
        
        conn.commit()
        
        # Извлекаем диалоги
        print("Извлекаю диалоги...")
        
        # Получаем сообщения
        try:
            cursor.execute("SELECT * FROM messages LIMIT 10000")
            messages = cursor.fetchall()
            
            for msg in messages:
                if len(msg) >= 3 and msg[2]:  # Проверяем наличие текста
                    text = str(msg[2]).strip()
                    if len(text) > 10:  # Минимальная длина сообщения
                        dialogue_record = {
                            'id': f"alice_msg_{msg[0]}",
                            'text': clean_text(text),
                            'source': 'alice_dialogues',
                            'title': f"Диалог {msg[0]}",
                            'author': 'Alice Bot',
                            'language': 'ru',
                            'topic': 'dialogue',
                            'religion': 'none',
                            'text_length': len(text),
                            'word_count': len(text.split()),
                            'context_type': 'dialogue'
                        }
                        dialogues.append(dialogue_record)
        except Exception as e:
            print(f"Ошибка при извлечении сообщений: {e}")
        
        # Получаем сценарии
        try:
            cursor.execute("SELECT * FROM scripts LIMIT 5000")
            scripts = cursor.fetchall()
            
            for script in scripts:
                if len(script) >= 2 and script[1]:  # Проверяем наличие текста сценария
                    text = str(script[1]).strip()
                    if len(text) > 10:
                        script_record = {
                            'id': f"alice_script_{script[0]}",
                            'text': clean_text(text),
                            'source': 'alice_scripts',
                            'title': f"Сценарий {script[0]}",
                            'author': 'Alice Bot',
                            'language': 'ru',
                            'topic': 'dialogue',
                            'religion': 'none',
                            'text_length': len(text),
                            'word_count': len(text.split()),
                            'context_type': 'script'
                        }
                        dialogues.append(script_record)
        except Exception as e:
            print(f"Ошибка при извлечении сценариев: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Ошибка при обработке SQL данных: {e}")
    finally:
        # Очищаем временные файлы
        if os.path.exists(temp_db):
            os.remove(temp_db)
        if os.path.exists("/tmp/alice_combined.sql"):
            os.remove("/tmp/alice_combined.sql")
    
    return dialogues

def create_combined_dataset(shridhar_books: List[Dict], alice_data: List[Dict]) -> DatasetDict:
    """Создание объединенного датасета"""
    
    # Объединяем все данные
    all_data = shridhar_books + alice_data
    
    print(f"Общее количество записей: {len(all_data)}")
    print(f"Книги Шридхара: {len(shridhar_books)}")
    print(f"Данные Alice: {len(alice_data)}")
    
    # Разделяем на train/validation/test (80/10/10)
    total_size = len(all_data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # Перемешиваем данные
    import random
    random.shuffle(all_data)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Создаем DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    return dataset_dict

def create_tokenizer_config():
    """Создание конфигурации токенизатора для русского языка"""
    tokenizer_config = {
        "vocab_size": 50000,
        "model_max_length": 8192,  # 8K контекст
        "special_tokens": [
            "<|spiritual|>",
            "<|dialogue|>", 
            "<|script|>",
            "<|end|>",
            "<|pad|>",
            "<|unk|>"
        ],
        "language": "ru",
        "tokenizer_type": "BPE",
        "lowercase": False,
        "strip_accents": False
    }
    return tokenizer_config

def create_training_config():
    """Создание конфигурации для обучения 8K модели"""
    training_config = {
        "model_config": {
            "vocab_size": 50000,
            "n_positions": 8192,      # 8K контекст
            "n_embd": 768,            # Размерность эмбеддингов
            "n_layer": 12,           # Количество слоев
            "n_head": 12,             # Количество голов внимания
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "activation_function": "gelu_new"
        },
        "training_args": {
            "per_device_train_batch_size": 1,      # Минимальный batch для 8K
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,     # Имитация batch_size=16
            "learning_rate": 1e-4,
            "num_train_epochs": 3,
            "max_steps": 10000,
            "warmup_steps": 500,
            "logging_steps": 100,
            "save_steps": 500,
            "eval_steps": 500,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "report_to": "tensorboard",
            "run_name": "shridhar_8k_training"
        },
        "optimization": {
            "use_mps": True,           # Использование GPU M4
            "mixed_precision": "fp16",
            "gradient_checkpointing": True,
            "flash_attention": True,
            "memory_efficient_attention": True
        }
    }
    return training_config

def main():
    """Основная функция подготовки датасета"""
    
    print("🚀 Начинаю подготовку объединенного датасета для обучения 8K модели...")
    
    # Пути к датасетам
    shridhar_path = "/Users/anton/proj/ai.nativemind.net/sridhar/datasets/shridhar_maharaj_books"
    alice_path = "/Users/anton/proj/ai.nativemind.net/sridhar/datasets/mozgach_alice_gift_sql_dataset"
    
    # Загружаем данные Шридхара Махараджа
    print("\n📚 Загружаю книги Шридхара Махараджа...")
    shridhar_books = load_shridhar_books(shridhar_path)
    print(f"✅ Загружено {len(shridhar_books)} книг")
    
    # Загружаем данные Alice
    print("\n🤖 Загружаю данные Alice...")
    alice_data = load_alice_sql_data(alice_path)
    print(f"✅ Загружено {len(alice_data)} диалогов/сценариев")
    
    # Создаем объединенный датасет
    print("\n🔄 Создаю объединенный датасет...")
    dataset_dict = create_combined_dataset(shridhar_books, alice_data)
    
    # Сохраняем датасет
    output_path = "/Users/anton/proj/ai.nativemind.net/sridhar/combined_shridhar_alice_dataset"
    print(f"\n💾 Сохраняю датасет в: {output_path}")
    dataset_dict.save_to_disk(output_path)
    
    # Создаем конфигурации
    print("\n⚙️ Создаю конфигурации...")
    
    # Конфигурация токенизатора
    tokenizer_config = create_tokenizer_config()
    with open(f"{output_path}/tokenizer_config.json", 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    # Конфигурация обучения
    training_config = create_training_config()
    with open(f"{output_path}/training_config.json", 'w', encoding='utf-8') as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    # Создаем README
    readme_content = f"""# Объединенный датасет для обучения модели Шридхара Махараджа

## Описание
Объединенный датасет содержит:
- {len(shridhar_books)} книг Шридхара Махараджа (духовная тематика)
- {len(alice_data)} диалогов/сценариев Alice (разговорная тематика)

## Структура
- **Train**: {len(dataset_dict['train'])} записей
- **Validation**: {len(dataset_dict['validation'])} записей  
- **Test**: {len(dataset_dict['test'])} записей

## Конфигурация
- **Максимальная длина контекста**: 8,192 токенов
- **Размер словаря**: 50,000 токенов
- **Архитектура**: GPT-подобная модель
- **Оптимизация**: MPS, FP16, Gradient Checkpointing

## Использование
```python
from datasets import load_from_disk

dataset = load_from_disk("{output_path}")
```
"""
    
    with open(f"{output_path}/README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Статистика
    total_text_length = sum(item['text_length'] for item in shridhar_books + alice_data)
    total_words = sum(item['word_count'] for item in shridhar_books + alice_data)
    
    print(f"\n📊 Статистика датасета:")
    print(f"   Общее количество записей: {len(shridhar_books + alice_data)}")
    print(f"   Книги Шридхара: {len(shridhar_books)}")
    print(f"   Данные Alice: {len(alice_data)}")
    print(f"   Общая длина текста: {total_text_length:,} символов")
    print(f"   Общее количество слов: {total_words:,}")
    print(f"   Train: {len(dataset_dict['train'])} записей")
    print(f"   Validation: {len(dataset_dict['validation'])} записей")
    print(f"   Test: {len(dataset_dict['test'])} записей")
    
    print(f"\n✅ Датасет успешно подготовлен!")
    print(f"📁 Путь: {output_path}")
    print(f"🎯 Готов к обучению модели с 8K контекстом")

if __name__ == "__main__":
    main()
