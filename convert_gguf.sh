#!/bin/bash
# Скрипт для конвертации модели в GGUF формат

echo "🔄 Конвертирую модель shridhar_8k в GGUF..."

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем зависимости
pip install llama-cpp-python

# Скачиваем llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Компилируем с CMake
mkdir build
cd build
cmake ..
make -j4

# Конвертируем модель
python ../convert_hf_to_gguf.py ../../shridhar_8k_merged --outfile ../../shridhar_8k.gguf --outtype f16

echo "✅ Конвертация завершена!"
