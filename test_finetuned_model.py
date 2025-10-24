#!/usr/bin/env python3
"""
Скрипт для тестирования обученной модели
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_finetuned_model():
    """Тестирование файнтюненной модели"""
    
    base_model_name = "nativemind/braindler_full_trained_model"
    lora_path = "./braindler_finetuned_8k_lora"
    
    print("Загружаю базовую модель...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Загружаю LoRA адаптеры...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Тестовые промпты
    test_prompts = [
        "Что такое бхакти-йога?",
        "Объясни философию Шрилы Шридхара Махараджа:",
        "Какова роль Кришны в вайшнавизме?",
        "Что означает преданное служение?",
        "Расскажи о концепции гуру в вайшнавизме:",
        "Что такое раса в философии гаудия-вайшнавов?",
        "Объясни разницу между дживой и Брахманом:",
        "Какова роль Шримати Радхарани в духовной практике?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Тест {i}/8")
        print(f"📝 Промпт: {prompt}")
        print(f"{'='*60}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Ответ: {response[len(prompt):]}")
        
        # Небольшая пауза между тестами
        import time
        time.sleep(1)

def test_context_length():
    """Тестирование длинного контекста"""
    print("\n" + "="*60)
    print("🔍 Тестирование длинного контекста (8K токенов)")
    print("="*60)
    
    base_model_name = "nativemind/braindler_full_trained_model"
    lora_path = "./braindler_finetuned_8k_lora"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Создаем длинный промпт для тестирования контекста
    long_prompt = """
    В философии гаудия-вайшнавов, как объясняет Шрила Шридхар Махарадж, 
    существует глубокая концепция бхакти-йоги, которая представляет собой 
    путь преданного служения Всевышнему Господу. Эта философия основана 
    на учении Шри Чайтаньи Махапрабху и его последователей, включая 
    Шрилу Рупу Госвами, Шрилу Санатану Госвами и других великих ачарьев.
    
    Основные принципы включают в себя:
    1. Концепцию дживы как вечной частицы Бога
    2. Роль гуру в духовном развитии
    3. Важность святого имени
    4. Служение в настроении преданности
    5. Понимание расы и бхавы
    
    Расскажи подробно о каждом из этих принципов и их практическом применении:
    """
    
    print(f"📝 Длинный промпт ({len(long_prompt)} символов):")
    print(long_prompt[:200] + "...")
    
    inputs = tokenizer(long_prompt, return_tensors="pt")
    print(f"🔢 Количество токенов в промпте: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 Ответ модели:")
    print(response[len(long_prompt):])

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование файнтюненной модели Braindler с контекстом 8K")
    print("="*70)
    
    try:
        # Основные тесты
        test_finetuned_model()
        
        # Тест длинного контекста
        test_context_length()
        
        print("\n✅ Все тесты завершены успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        print("Убедитесь, что модель была обучена и сохранена корректно.")

if __name__ == "__main__":
    main()
