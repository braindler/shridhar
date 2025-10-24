#!/usr/bin/env python3
"""
Скрипт для загрузки датасета shridhar_maharaj_books в Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from datasets import load_from_disk

def upload_dataset():
    """Загрузка датасета в Hugging Face Hub"""
    
    # Инициализация API
    api = HfApi()
    
    # Настройки репозитория
    repo_id = "nativemind/shridhar_maharaj_books"
    dataset_path = "/Users/anton/proj/ai.nativemind.net/sridhar/shridhar_maharaj_books_dataset"
    
    print(f"Создаю репозиторий: {repo_id}")
    
    try:
        # Создаем репозиторий (если не существует)
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,  # Публичный датасет
            exist_ok=True
        )
        print("Репозиторий создан успешно")
        
    except Exception as e:
        print(f"Ошибка при создании репозитория: {e}")
        return
    
    print("Загружаю датасет...")
    
    try:
        # Загружаем датасет
        dataset = load_from_disk(dataset_path)
        
        # Загружаем в Hub
        dataset.push_to_hub(
            repo_id=repo_id,
            private=False
        )
        
        print(f"Датасет успешно загружен в: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")

def main():
    """Основная функция"""
    print("Начинаю загрузку датасета в Hugging Face Hub...")
    
    # Проверяем, что пользователь авторизован
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Авторизован как: {user['name']}")
    except Exception as e:
        print("Ошибка авторизации. Пожалуйста, выполните:")
        print("huggingface-cli login")
        return
    
    upload_dataset()

if __name__ == "__main__":
    main()
