#!/usr/bin/env python3
"""
Скрипт для файнтюнинга модели braindler_full_trained_model
с контекстом 8K на датасетах: alpaca_data, mozgach_alice_gift_sql_dataset, 
mozgach_alpaca_gift, shridhar_maharaj_books
"""

import torch
import os
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, concatenate_datasets
import wandb
from accelerate import Accelerator
import deepspeed
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

class BraindlerFinetuner:
    def __init__(self, model_name="nativemind/braindler_full_trained_model", max_length=8192):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Настройка модели и токенизатора"""
        print(f"Загружаю модель: {self.model_name}")
        
        # Настройка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Конфигурация для 8K контекста
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Расширяем контекст до 8K
        self.extend_context_length()
        
        print(f"Модель загружена. Максимальная длина: {self.max_length}")
        
    def extend_context_length(self):
        """Расширение контекста модели до 8K"""
        config = self.model.config
        
        # Изменяем максимальную длину позиций
        if hasattr(config, 'n_positions'):
            config.n_positions = self.max_length
        elif hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = self.max_length
        
        # Настройка RoPE для длинного контекста
        if hasattr(config, 'rope_theta'):
            config.rope_theta = 10000.0
        
        print(f"Контекст расширен до {self.max_length} токенов")
        
    def load_datasets(self):
        """Загрузка всех датасетов"""
        datasets = []
        
        # Список датасетов для загрузки
        dataset_configs = [
            "nativemind/alpaca_data",
            "nativemind/mozgach_alice_gift_sql_dataset", 
            "nativemind/mozgach_alpaca_gift",
            "nativemind/shridhar_maharaj_books"
        ]
        
        for dataset_name in dataset_configs:
            try:
                print(f"Загружаю датасет: {dataset_name}")
                dataset = load_dataset(dataset_name)
                
                # Обрабатываем разные структуры датасетов
                if 'train' in dataset:
                    if len(dataset) == 1:  # Только train
                        datasets.append(dataset['train'])
                    else:  # Есть train/validation/test
                        # Объединяем все разделы
                        combined = concatenate_datasets([
                            dataset['train'], 
                            dataset.get('validation', dataset['train']),
                            dataset.get('test', dataset['train'])
                        ])
                        datasets.append(combined)
                else:
                    # Если структура нестандартная, берем первый доступный ключ
                    first_key = list(dataset.keys())[0]
                    datasets.append(dataset[first_key])
                    
                print(f"✅ {dataset_name}: {len(datasets[-1])} записей")
                
            except Exception as e:
                print(f"❌ Ошибка загрузки {dataset_name}: {e}")
                continue
        
        if not datasets:
            raise ValueError("Не удалось загрузить ни одного датасета")
        
        # Объединяем все датасеты
        combined_dataset = concatenate_datasets(datasets)
        print(f"📊 Общий размер объединенного датасета: {len(combined_dataset)}")
        
        return combined_dataset
        
    def prepare_dataset_for_training(self, dataset):
        """Подготовка датасета для обучения"""
        
        def tokenize_function(examples):
            # Собираем все текстовые поля
            texts = []
            
            for key in examples.keys():
                if isinstance(examples[key][0], str):
                    # Если это текстовое поле, добавляем его
                    texts.extend(examples[key])
                elif isinstance(examples[key][0], list):
                    # Если это список строк, объединяем их
                    for item in examples[key]:
                        if isinstance(item, list):
                            texts.append(' '.join(item))
                        else:
                            texts.append(str(item))
            
            # Токенизируем с учетом максимальной длины
            return self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None  # Убираем return_tensors для правильной обработки
            )
        
        print("Токенизирую датасет...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
        
    def setup_lora(self):
        """Настройка LoRA для эффективного файнтюнинга"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc", "c_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA настроен для эффективного файнтюнинга")
        
    def train(self):
        """Основная функция обучения"""
        
        # Инициализация W&B (опционально)
        try:
            wandb.init(
                project="braindler-8k-finetune",
                name="braindler_8k_experiment",
                config={
                    "model": self.model_name,
                    "max_length": self.max_length,
                    "datasets": [
                        "alpaca_data",
                        "mozgach_alice_gift_sql_dataset", 
                        "mozgach_alpaca_gift",
                        "shridhar_maharaj_books"
                    ]
                }
            )
            print("✅ W&B инициализирован")
        except Exception as e:
            print(f"⚠️ W&B не настроен: {e}")
            print("Продолжаю без мониторинга...")
        
        # Настройка модели и токенизатора
        self.setup_model_and_tokenizer()
        
        # Настройка LoRA
        self.setup_lora()
        
        # Загрузка датасетов
        dataset = self.load_datasets()
        
        # Подготовка датасета
        tokenized_dataset = self.prepare_dataset_for_training(dataset)
        
        # Разделение на train/validation
        train_size = int(0.9 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        print(f"📈 Train: {len(train_dataset)} записей")
        print(f"📈 Validation: {len(eval_dataset)} записей")
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir="./braindler_finetuned_8k",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=10,
            eval_steps=500,
            save_steps=1000,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            run_name="braindler_8k_finetune",
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Инициализация Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("🚀 Начинаю обучение...")
        trainer.train()
        
        print("💾 Сохраняю модель...")
        trainer.save_model()
        self.tokenizer.save_pretrained("./braindler_finetuned_8k")
        
        # Сохраняем LoRA адаптеры
        self.model.save_pretrained("./braindler_finetuned_8k_lora")
        
        print("✅ Обучение завершено!")
        
        # Тестирование модели
        self.test_model()
        
    def test_model(self):
        """Тестирование обученной модели"""
        print("🧪 Тестирую модель...")
        
        test_prompts = [
            "Расскажи о философии вайшнавизма:",
            "Что такое бхакти-йога?",
            "Объясни концепцию Кришны:",
            "Какова роль гуру в духовной практике?"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Промпт: {prompt}")
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"🤖 Ответ: {response[len(prompt):]}")

def main():
    """Основная функция"""
    print("🚀 Запуск файнтюнинга Braindler с контекстом 8K")
    
    # Создаем экземпляр файнтюнера
    finetuner = BraindlerFinetuner(max_length=8192)
    
    # Запускаем обучение
    finetuner.train()

if __name__ == "__main__":
    main()
