import json

# Путь к файлу с документами
input_file = "documents/documents.txt"
# Путь для сохранения passages.jsonl
output_file = "documents/passages.jsonl"

# Чтение документов из файла
with open(input_file, "r", encoding="utf-8") as f:
    documents = f.readlines()

# Создание passages.jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(documents):
        # Удаляем лишние пробелы и символы новой строки
        doc = doc.strip()
        # Формируем JSON-объект
        passage = {"id": str(idx), "text": doc}
        # Записываем в файл
        f.write(json.dumps(passage, ensure_ascii=False) + "\n")