#!/usr/bin/env python3
"""
Обучение модели Шридхара Махараджа с 8K контекстом на MacBook Pro M4
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

class ShridharModelConfig:
    """Конфигурация модели для 8K контекста"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.n_positions = 8192  # 8K контекст
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

class ShridharTokenizer:
    """Токенизатор для русского языка"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|spiritual|>": 0,
            "<|dialogue|>": 1,
            "<|script|>": 2,
            "<|end|>": 3,
            "<|pad|>": 4,
            "<|unk|>": 5
        }
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Построение словаря"""
        # Добавляем специальные токены
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Простая реализация BPE для русского языка
        # В реальном проекте здесь был бы полный BPE токенизатор
        current_id = len(self.special_tokens)
        
        # Базовые символы
        for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
            if current_id < self.vocab_size:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Добавляем пробелы и знаки препинания
        for char in " .,!?;:-()[]{}":
            if current_id < self.vocab_size:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
    
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в токены"""
        tokens = []
        for char in text.lower():
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.special_tokens["<|unk|>"])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодирование токенов в текст"""
        text = ""
        for token_id in token_ids:
            if token_id in self.id_to_token:
                text += self.id_to_token[token_id]
        return text

class ShridharDataset:
    """Датасет для обучения модели"""
    
    def __init__(self, dataset_path: str, tokenizer: ShridharTokenizer, max_length: int = 8192):
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

def create_model(config: ShridharModelConfig) -> GPT2LMHeadModel:
    """Создание модели GPT-2 для 8K контекста"""
    
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

def setup_training_environment():
    """Настройка среды обучения для M4"""
    
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

def create_training_arguments():
    """Создание аргументов обучения для 8K контекста"""
    
    return TrainingArguments(
        output_dir="./shridhar_8k_model",
        per_device_train_batch_size=1,        # Минимальный batch для 8K
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,       # Имитация batch_size=16
        learning_rate=1e-4,
        num_train_epochs=3,
        max_steps=10000,
        warmup_steps=500,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,                          # MPS не поддерживает fp16, используем fp32
        gradient_checkpointing=True,         # Экономия памяти
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name="shridhar_8k_training",
        save_total_limit=3,                 # Ограничиваем количество сохранений
        prediction_loss_only=True,
        dataloader_drop_last=True
    )

def main():
    """Основная функция обучения"""
    
    logger.info("🚀 Начинаю обучение модели Шридхара Махараджа с 8K контекстом...")
    
    # Настройка среды
    device = setup_training_environment()
    logger.info(f"🖥️ Устройство: {device}")
    
    # Загрузка конфигурации
    config_path = "/Users/anton/proj/ai.nativemind.net/sridhar/combined_shridhar_alice_dataset/training_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        training_config = json.load(f)
    
    # Создание конфигурации модели
    model_config = ShridharModelConfig()
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
    model = create_model(model_config)
    model.to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Параметры модели: {total_params:,} (обучаемых: {trainable_params:,})")
    
    # Загрузка датасета
    dataset_path = "/Users/anton/proj/ai.nativemind.net/sridhar/combined_shridhar_alice_dataset"
    logger.info(f"📚 Загружаю датасет из: {dataset_path}")
    
    train_dataset = ShridharDataset(dataset_path, tokenizer, max_length=model_config.n_positions)
    logger.info(f"📖 Размер обучающего датасета: {len(train_dataset)}")
    
    # Создание DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, не MLM
        pad_to_multiple_of=8
    )
    
    # Создание аргументов обучения
    training_args = create_training_arguments()
    logger.info(f"⚙️ Аргументы обучения: batch_size={training_args.per_device_train_batch_size}, "
                f"gradient_accumulation={training_args.gradient_accumulation_steps}")
    
    # Создание валидационного датасета
    val_dataset = ShridharDataset(dataset_path, tokenizer, max_length=model_config.n_positions)
    val_dataset.dataset = val_dataset.dataset['validation']  # Используем validation split
    
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
            # Для MPS нет прямого способа получить использование памяти
            logger.info("💾 MPS активен - использует GPU M4")
        else:
            logger.info("💾 CPU режим")
    
    # Запуск обучения
    logger.info("🎯 Начинаю обучение...")
    log_memory_usage()
    
    try:
        trainer.train()
        logger.info("✅ Обучение завершено успешно!")
        
        # Сохранение модели
        logger.info("💾 Сохраняю модель...")
        trainer.save_model()
        
        # Сохранение токенизатора
        tokenizer_config = {
            "vocab_size": model_config.vocab_size,
            "special_tokens": tokenizer.special_tokens,
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": tokenizer.id_to_token
        }
        
        with open("./shridhar_8k_model/tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        logger.info("🎉 Модель успешно сохранена в ./shridhar_8k_model/")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обучении: {e}")
        raise

if __name__ == "__main__":
    main()
