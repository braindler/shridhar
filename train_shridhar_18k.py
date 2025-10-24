#!/usr/bin/env python3
"""
Обучение модели Шридхара Махараджа с 18K контекстом на MacBook Pro M4
Оптимизированная версия для максимального использования памяти
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel
)
from datasets import load_from_disk
import numpy as np
from typing import Dict, List, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShridharModelConfig18K:
    """Конфигурация модели для 18K контекста"""
    
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 размер словаря
        self.n_positions = 18432  # 18K контекст
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.attn_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.activation_function = "gelu_new"
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

class ShridharDataset18K:
    """Датасет для обучения модели с 18K контекстом"""
    
    def __init__(self, dataset_path: str, tokenizer, max_length: int = 18432):
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset['train'])
    
    def __getitem__(self, idx):
        item = self.dataset['train'][idx]
        text = item['text']
        
        # Добавляем специальные токены в зависимости от типа контента
        if item.get('context_type') == 'spiritual_text':
            text = f"<|spiritual|>{text}<|end|>"
        elif item.get('context_type') == 'dialogue':
            text = f"<|dialogue|>{text}<|end|>"
        elif item.get('context_type') == 'script':
            text = f"<|script|>{text}<|end|>"
        
        # Токенизируем с помощью GPT-2 токенизатора
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }

def create_model_18k(config: ShridharModelConfig18K) -> GPT2LMHeadModel:
    """Создание модели GPT-2 для 18K контекста"""
    
    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        attn_pdrop=config.attn_pdrop,
        embd_pdrop=config.embd_pdrop,
        resid_pdrop=config.resid_pdrop,
        activation_function=config.activation_function,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id
    )
    
    model = GPT2LMHeadModel(model_config)
    return model

def setup_training_environment_18k():
    """Настройка среды обучения для M4 с 18K контекстом"""
    
    # Проверяем доступность MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ Используется MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ MPS недоступен, используется CPU")
    
    # Настройки для оптимизации памяти
    if device.type == "mps":
        # Для MPS нет прямого аналога empty_cache, но можно очистить кэш Python
        import gc
        gc.collect()
        logger.info("🧹 Очистка памяти выполнена")
    
    return device

def create_training_arguments_18k():
    """Создание аргументов обучения для 18K контекста"""
    
    return TrainingArguments(
        output_dir="./shridhar_18k_model",
        per_device_train_batch_size=1,        # Минимальный batch для 18K
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,       # Имитация batch_size=32
        learning_rate=5e-5,                   # Меньший learning rate для стабильности
        num_train_epochs=3,
        max_steps=15000,                      # Больше шагов для 18K
        warmup_steps=1000,                    # Больше warmup
        logging_steps=200,                    # Реже логирование
        save_steps=1000,                      # Реже сохранение
        eval_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,                          # MPS не поддерживает fp16
        gradient_checkpointing=True,         # Критично для экономии памяти
        dataloader_pin_memory=False,         # Отключаем для экономии памяти
        dataloader_num_workers=2,            # Меньше воркеров
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name="shridhar_18k_training",
        save_total_limit=2,                 # Ограничиваем количество сохранений
        prediction_loss_only=True,
        dataloader_drop_last=True,
        # Дополнительные оптимизации для 18K
        dataloader_prefetch_factor=2,        # Меньше prefetch
        max_grad_norm=1.0,                  # Gradient clipping
        weight_decay=0.01,                  # Weight decay
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8
    )

def monitor_memory_usage():
    """Мониторинг использования памяти"""
    import psutil
    
    # Получаем информацию о памяти
    memory = psutil.virtual_memory()
    logger.info(f"💾 Память: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    
    # Предупреждение при высоком использовании
    if memory.percent > 90:
        logger.warning("⚠️ Высокое использование памяти! Возможен OOM.")
    elif memory.percent > 80:
        logger.warning("⚠️ Умеренно высокое использование памяти.")
    else:
        logger.info("✅ Использование памяти в норме.")

def main():
    """Основная функция обучения с 18K контекстом"""
    
    logger.info("🚀 Начинаю обучение модели Шридхара Махараджа с 18K контекстом...")
    
    # Настройка среды
    device = setup_training_environment_18k()
    logger.info(f"🖥️ Устройство: {device}")
    
    # Мониторинг памяти
    monitor_memory_usage()
    
    # Создание конфигурации модели
    model_config = ShridharModelConfig18K()
    logger.info(f"📐 Конфигурация модели: {model_config.n_positions} токенов, {model_config.n_layer} слоев")
    
    # Создание токенизатора (используем GPT-2 токенизатор)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"🔤 Токенизатор: {len(tokenizer)} токенов")
    
    # Обновляем размер словаря в конфигурации
    model_config.vocab_size = len(tokenizer)
    
    # Создание модели
    logger.info("🧠 Создаю модель...")
    model = create_model_18k(model_config)
    model.to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Параметры модели: {total_params:,} (обучаемых: {trainable_params:,})")
    
    # Мониторинг памяти после создания модели
    monitor_memory_usage()
    
    # Загрузка датасета
    dataset_path = "/Users/anton/proj/ai.nativemind.net/sridhar/combined_shridhar_alice_dataset"
    logger.info(f"📚 Загружаю датасет из: {dataset_path}")
    
    train_dataset = ShridharDataset18K(dataset_path, tokenizer, max_length=model_config.n_positions)
    logger.info(f"📖 Размер обучающего датасета: {len(train_dataset)}")
    
    # Создание валидационного датасета
    val_dataset = ShridharDataset18K(dataset_path, tokenizer, max_length=model_config.n_positions)
    val_dataset.dataset = val_dataset.dataset['validation']  # Используем validation split
    
    # Создание DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, не MLM
        pad_to_multiple_of=8
    )
    
    # Создание аргументов обучения
    training_args = create_training_arguments_18k()
    logger.info(f"⚙️ Аргументы обучения: batch_size={training_args.per_device_train_batch_size}, "
                f"gradient_accumulation={training_args.gradient_accumulation_steps}")
    
    # Мониторинг памяти перед обучением
    monitor_memory_usage()
    
    # Создание тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Мониторинг памяти
    def log_memory_usage():
        if torch.backends.mps.is_available():
            logger.info("💾 MPS активен - использует GPU M4")
        else:
            logger.info("💾 CPU режим")
        monitor_memory_usage()
    
    # Запуск обучения
    logger.info("🎯 Начинаю обучение с 18K контекстом...")
    log_memory_usage()
    
    try:
        # Мониторинг памяти каждые 100 шагов
        original_train = trainer.train
        
        def monitored_train(*args, **kwargs):
            result = original_train(*args, **kwargs)
            monitor_memory_usage()
            return result
        
        trainer.train = monitored_train
        
        trainer.train()
        logger.info("✅ Обучение завершено успешно!")
        
        # Сохранение модели
        logger.info("💾 Сохраняю модель...")
        trainer.save_model()
        
        # Сохранение токенизатора
        tokenizer_config = {
            "vocab_size": model_config.vocab_size,
            "model_max_length": model_config.n_positions,
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token
            }
        }
        
        with open("./shridhar_18k_model/tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        logger.info("🎉 Модель 18K успешно сохранена в ./shridhar_18k_model/")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обучении: {e}")
        monitor_memory_usage()
        raise

if __name__ == "__main__":
    main()

