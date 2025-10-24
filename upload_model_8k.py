#!/usr/bin/env python3
"""
Скрипт для загрузки обученной модели shridhar_8k на Hugging Face Hub
"""

import os
import json
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def upload_model():
    """Загрузка обученной модели на Hugging Face Hub"""
    
    # Инициализация API
    api = HfApi()
    
    # Настройки репозитория
    repo_id = "nativemind/shridhar_8k"
    model_path = "./braindler_finetuned_8k"
    lora_path = "./braindler_finetuned_8k_lora"
    
    print(f"Создаю репозиторий: {repo_id}")
    
    try:
        # Создаем репозиторий (если не существует)
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,  # Публичная модель
            exist_ok=True
        )
        print("✅ Репозиторий создан успешно")
        
    except Exception as e:
        print(f"❌ Ошибка при создании репозитория: {e}")
        return
    
    print("📤 Загружаю модель...")
    
    try:
        # Загружаем базовую модель и LoRA адаптеры
        base_model_name = "nativemind/braindler_full_trained_model"
        
        print("Загружаю базовую модель...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="cpu"  # Загружаем на CPU для загрузки
        )
        
        print("Загружаю LoRA адаптеры...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        # Объединяем LoRA с базовой моделью
        print("Объединяю LoRA с базовой моделью...")
        merged_model = model.merge_and_unload()
        
        # Загружаем в Hub
        print("Загружаю объединенную модель в Hub...")
        merged_model.push_to_hub(
            repo_id=repo_id,
            private=False
        )
        
        # Загружаем токенизатор
        print("Загружаю токенизатор...")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            private=False
        )
        
        # Создаем README.md для модели
        create_model_card(repo_id)
        
        print(f"✅ Модель успешно загружена в: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")

def create_model_card(repo_id):
    """Создание README.md для модели"""
    
    card_content = f"""---
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
- lora
- finetuned
base_model: nativemind/braindler_full_trained_model
pipeline_tag: text-generation
---

# Shridhar 8K - Философия вайшнавизма

## Описание

Это файнтюненная версия модели `braindler_full_trained_model` с расширенным контекстом до 8K токенов, специально обученная на духовных текстах Шрилы Шридхара Махараджа.

## Особенности

- **Контекст**: 8192 токенов (8K)
- **Архитектура**: GPT-2 с LoRA адаптерами
- **Язык**: русский
- **Тематика**: философия вайшнавизма, кришнаизм, духовность
- **Обучение**: LoRA файнтюнинг на датасете `shridhar_maharaj_books`

## Использование

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Загрузка модели
model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Генерация текста
prompt = "Расскажи о философии вайшнавизма:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Датасет обучения

Модель обучена на датасете `nativemind/shridhar_maharaj_books`, содержащем:
- 15 книг Шрилы Шридхара Махараджа
- 2.2MB текста
- 313K слов
- Философские и духовные тексты на русском языке

## Технические детали

- **Base Model**: nativemind/braindler_full_trained_model
- **LoRA Rank**: 4
- **LoRA Alpha**: 8
- **Target Modules**: c_attn, c_proj
- **Learning Rate**: 1e-5
- **Epochs**: 1
- **Gradient Checkpointing**: Enabled

## Авторские права

Все тексты принадлежат Шри Чайтанья Сарасват Матху и распространяются в соответствии с их лицензией.

## Контакт

Для вопросов по модели обращайтесь к создателю репозитория.
"""
    
    # Сохраняем README.md локально
    with open("./model_readme.md", 'w', encoding='utf-8') as f:
        f.write(card_content)
    
    print("✅ README.md создан")

def main():
    """Основная функция"""
    print("🚀 Загрузка модели shridhar_8k на Hugging Face Hub")
    
    # Проверяем, что пользователь авторизован
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Авторизован как: {user['name']}")
    except Exception as e:
        print("Ошибка авторизации. Пожалуйста, выполните:")
        print("huggingface-cli login")
        return
    
    upload_model()

if __name__ == "__main__":
    main()
