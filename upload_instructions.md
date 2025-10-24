# Инструкция по загрузке датасета в Hugging Face Hub

## Шаги для загрузки

### 1. Авторизация в Hugging Face

Сначала необходимо авторизоваться в Hugging Face:

```bash
# Установите huggingface-cli если еще не установлен
pip install huggingface_hub

# Авторизуйтесь
huggingface-cli login
```

Введите ваш токен доступа из https://huggingface.co/settings/tokens

### 2. Загрузка датасета

Запустите скрипт загрузки:

```bash
cd /Users/anton/proj/ai.nativemind.net/sridhar
source venv/bin/activate
python upload_to_huggingface.py
```

### 3. Альтернативный способ - ручная загрузка

Если автоматическая загрузка не работает, можно загрузить вручную:

```python
from datasets import load_from_disk
from huggingface_hub import HfApi

# Загружаем датасет
dataset = load_from_disk("/Users/anton/proj/ai.nativemind.net/sridhar/shridhar_maharaj_books_dataset")

# Загружаем в Hub
dataset.push_to_hub("your-username/shridhar_maharaj_books")
```

### 4. Проверка загрузки

После загрузки датасет будет доступен по адресу:
https://huggingface.co/datasets/your-username/shridhar_maharaj_books

## Структура датасета

Датасет содержит:
- **15 книг** Шрилы Шридхара Махараджа
- **2.2M символов** текста
- **313K слов**
- Разделение: train/validation/test (80/10/10)

## Использование датасета

```python
from datasets import load_dataset

# Загрузка датасета
dataset = load_dataset("your-username/shridhar_maharaj_books")

# Просмотр структуры
print(dataset)

# Доступ к данным
train_data = dataset['train']
print(f"Количество книг в train: {len(train_data)}")
print(f"Первая книга: {train_data[0]['title']}")
```
