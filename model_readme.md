---
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
model_name = "nativemind/shridhar_8k"
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
