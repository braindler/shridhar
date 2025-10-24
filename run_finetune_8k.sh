#!/bin/bash
# run_finetune_8k.sh

echo "🚀 Запуск файнтюнинга Braindler с контекстом 8K"

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем дополнительные зависимости
echo "📦 Устанавливаю дополнительные зависимости..."
pip install peft bitsandbytes wandb

# Настраиваем W&B
export WANDB_PROJECT="braindler-8k-finetune"
export WANDB_MODE="online"

# Проверяем доступность GPU
echo "🔍 Проверяю доступность GPU..."
nvidia-smi

# Запускаем обучение с DeepSpeed
echo "🚀 Запускаю обучение с DeepSpeed..."
deepspeed --num_gpus=1 finetune_braindler_8k.py

echo "✅ Обучение завершено!"
