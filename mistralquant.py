from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

# Настройка квантования (4-битное квантование)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Включаем 4-битное квантование
    bnb_4bit_compute_dtype=torch.float16,  # Используем FP16 для вычислений
    bnb_4bit_quant_type="nf4",  # Тип квантования (nf4 или fp4)
    bnb_4bit_use_double_quant=True,  # Двойное квантование для большей эффективности
)

# Загрузка модели с квантованием и оффлоадингом на CPU
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Автоматическое распределение между GPU и CPU
    quantization_config=quantization_config,  # Передаем конфигурацию квантования
    llm_int8_enable_fp32_cpu_offload=True,  # Включаем оффлоадинг на CPU
)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Создание pipeline для генерации текста
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"  # Автоматическое распределение между GPU и CPU
)

# Сообщения для чата
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Форматирование сообщений для модели Mistral
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)

# Генерация ответа
response = chatbot(formatted_input, max_length=100, do_sample=True, temperature=0.7)

# Вывод результата
print(response[0]['generated_text'])