#!/usr/bin/env python3
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
        stop=["</s>", "

"]
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
        print(f"\n{'='*60}")
        print(f"📝 Промпт: {prompt}")
        print(f"{'='*60}")
        
        response = generate_text(llm, prompt)
        print(f"🤖 Ответ: {response}")

if __name__ == "__main__":
    main()
