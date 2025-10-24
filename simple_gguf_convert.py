#!/usr/bin/env python3
"""
Простая конвертация модели в GGUF формат
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import subprocess

def convert_to_gguf():
    """Конвертация модели в GGUF формат"""
    print("🔄 Конвертирую модель shridhar_8k в GGUF...")
    
    # Загружаем модель
    base_model_name = "nativemind/braindler_full_trained_model"
    lora_path = "./braindler_finetuned_8k"
    
    print("Загружаю базовую модель...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    print("Загружаю LoRA адаптеры...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Объединяю LoRA с базовой моделью...")
    merged_model = model.merge_and_unload()
    
    # Сохраняем объединенную модель
    output_dir = "./shridhar_8k_merged"
    print(f"Сохраняю объединенную модель в {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Создаем конфигурацию для GGUF
    config = {
        "model_type": "gpt2",
        "vocab_size": merged_model.config.vocab_size,
        "n_positions": 8192,  # 8K контекст
        "n_embd": merged_model.config.n_embd,
        "n_layer": merged_model.config.n_layer,
        "n_head": merged_model.config.n_head,
        "activation_function": merged_model.config.activation_function,
        "bos_token_id": merged_model.config.bos_token_id,
        "eos_token_id": merged_model.config.eos_token_id,
        "pad_token_id": merged_model.config.pad_token_id
    }
    
    # Сохраняем конфигурацию
    import json
    with open(f"{output_dir}/gguf_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Модель готова для конвертации в GGUF")
    print(f"📁 Объединенная модель сохранена в: {output_dir}")
    print("📋 Для полной конвертации в GGUF используйте llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("   cd llama.cpp && make")
    print("   python convert_hf_to_gguf.py ../shridhar_8k_merged --outfile ../shridhar_8k.gguf --outtype f16")
    
    return output_dir

def create_usage_example():
    """Создание примера использования GGUF модели"""
    usage_example = '''#!/usr/bin/env python3
"""
Пример использования GGUF модели shridhar_8k
"""

from llama_cpp import Llama

def main():
    """Основная функция"""
    print("🚀 Использование GGUF модели shridhar_8k")
    
    # Загружаем GGUF модель
    print("Загружаю GGUF модель...")
    llm = Llama(
        model_path="./shridhar_8k.gguf",
        n_ctx=8192,  # 8K контекст
        n_threads=4,
        verbose=False
    )
    
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
        
        response = llm(
            prompt,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "\\n\\n"]
        )
        
        print(f"🤖 Ответ: {response['choices'][0]['text']}")

if __name__ == "__main__":
    main()
'''
    
    with open("./use_gguf_example.py", "w") as f:
        f.write(usage_example)
    
    print("✅ Пример использования GGUF создан: use_gguf_example.py")

def main():
    """Основная функция"""
    print("🚀 Подготовка модели shridhar_8k для GGUF конвертации")
    
    # Конвертируем модель
    model_path = convert_to_gguf()
    
    # Создаем пример использования
    create_usage_example()
    
    print("\\n📋 Следующие шаги:")
    print("1. Установите llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
    print("2. Скомпилируйте: cd llama.cpp && make")
    print("3. Конвертируйте: python convert_hf_to_gguf.py ../shridhar_8k_merged --outfile ../shridhar_8k.gguf --outtype f16")
    print("4. Используйте: python use_gguf_example.py")
    
    print("\\n✅ Модель готова для конвертации в GGUF!")

if __name__ == "__main__":
    main()
