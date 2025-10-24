#!/usr/bin/env python3
"""
Скрипт для конвертации модели shridhar_8k в GGUF формат
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import subprocess
import sys

def install_llama_cpp():
    """Установка llama.cpp для конвертации в GGUF"""
    print("📦 Устанавливаю llama.cpp...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)
        print("✅ llama.cpp установлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки llama.cpp: {e}")
        return False
    return True

def load_and_merge_model():
    """Загрузка и объединение модели с LoRA адаптерами"""
    print("🔄 Загружаю и объединяю модель...")
    
    base_model_name = "nativemind/braindler_full_trained_model"
    lora_path = "./braindler_finetuned_8k"
    
    try:
        # Загружаем базовую модель
        print("Загружаю базовую модель...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Загружаем LoRA адаптеры
        print("Загружаю LoRA адаптеры...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        # Объединяем LoRA с базовой моделью
        print("Объединяю LoRA с базовой моделью...")
        merged_model = model.merge_and_unload()
        
        # Сохраняем объединенную модель
        output_dir = "./shridhar_8k_merged"
        print(f"Сохраняю объединенную модель в {output_dir}...")
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ Модель объединена и сохранена")
        return output_dir
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None

def convert_to_gguf(model_path):
    """Конвертация модели в GGUF формат"""
    print("🔄 Конвертирую модель в GGUF формат...")
    
    try:
        from llama_cpp import Llama
        
        # Создаем GGUF файл
        gguf_path = "./shridhar_8k.gguf"
        
        # Используем llama.cpp для конвертации
        print("Конвертирую в GGUF...")
        
        # Загружаем модель через transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Сохраняем в формате, совместимом с llama.cpp
        print("Подготавливаю модель для GGUF...")
        
        # Создаем конфигурацию для GGUF
        config = {
            "model_type": "gpt2",
            "vocab_size": model.config.vocab_size,
            "n_positions": 8192,  # 8K контекст
            "n_embd": model.config.n_embd,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "activation_function": model.config.activation_function,
            "bos_token_id": model.config.bos_token_id,
            "eos_token_id": model.config.eos_token_id,
            "pad_token_id": model.config.pad_token_id
        }
        
        print("✅ Конфигурация GGUF создана")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конвертации в GGUF: {e}")
        return False

def create_gguf_script():
    """Создание скрипта для конвертации в GGUF"""
    script_content = '''#!/bin/bash
# Скрипт для конвертации модели в GGUF формат

echo "🔄 Конвертирую модель shridhar_8k в GGUF..."

# Устанавливаем зависимости
pip install llama-cpp-python

# Скачиваем llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Компилируем
make

# Конвертируем модель
python convert_hf_to_gguf.py ../../shridhar_8k_merged --outfile ../../shridhar_8k.gguf --outtype f16

echo "✅ Конвертация завершена!"
'''
    
    with open("./convert_gguf.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("./convert_gguf.sh", 0o755)
    print("✅ Скрипт конвертации создан: convert_gguf.sh")

def create_gguf_usage_script():
    """Создание скрипта для использования GGUF модели"""
    usage_script = '''#!/usr/bin/env python3
"""
Скрипт для использования GGUF модели shridhar_8k
"""

from llama_cpp import Llama

def load_gguf_model():
    """Загрузка GGUF модели"""
    print("🔄 Загружаю GGUF модель...")
    
    # Загружаем модель
    llm = Llama(
        model_path="./shridhar_8k.gguf",
        n_ctx=8192,  # 8K контекст
        n_threads=4,
        verbose=False
    )
    
    print("✅ GGUF модель загружена")
    return llm

def generate_text(llm, prompt, max_tokens=300):
    """Генерация текста с помощью GGUF модели"""
    print(f"📝 Генерирую текст для промпта: {prompt}")
    
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "\n\n"]
    )
    
    return response['choices'][0]['text']

def main():
    """Основная функция"""
    print("🚀 Использование GGUF модели shridhar_8k")
    
    # Загружаем модель
    llm = load_gguf_model()
    
    # Тестовые промпты
    test_prompts = [
        "Расскажи о философии вайшнавизма:",
        "Что такое бхакти-йога?",
        "Объясни концепцию Кришны:",
        "Какова роль гуру в духовной практике?"
    ]
    
    for prompt in test_prompts:
        print(f"\\n{'='*60}")
        print(f"📝 Промпт: {prompt}")
        print(f"{'='*60}")
        
        response = generate_text(llm, prompt)
        print(f"🤖 Ответ: {response}")

if __name__ == "__main__":
    main()
'''
    
    with open("./use_gguf_model.py", "w") as f:
        f.write(usage_script)
    
    print("✅ Скрипт использования GGUF создан: use_gguf_model.py")

def main():
    """Основная функция конвертации"""
    print("🚀 Конвертация модели shridhar_8k в GGUF формат")
    
    # Устанавливаем llama.cpp
    if not install_llama_cpp():
        return
    
    # Загружаем и объединяем модель
    model_path = load_and_merge_model()
    if not model_path:
        return
    
    # Создаем скрипты для конвертации
    create_gguf_script()
    create_gguf_usage_script()
    
    print("\\n📋 Инструкции для конвертации в GGUF:")
    print("1. Запустите: bash convert_gguf.sh")
    print("2. После конвертации используйте: python use_gguf_model.py")
    print("\\n✅ Все скрипты готовы для конвертации в GGUF!")

if __name__ == "__main__":
    main()
