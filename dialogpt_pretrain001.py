from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Загрузка модели для создания эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Загрузка документов
with open("documents/documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Создание эмбеддингов для документов
document_embeddings = model.encode(documents, convert_to_tensor=False)

# Создание FAISS индекса
dimension = document_embeddings.shape[1]  # Размерность эмбеддингов
index = faiss.IndexFlatIP(dimension)  # Используем косинусное сходство (Inner Product)
index.add(document_embeddings.astype('float32'))

# Сохранение индекса
faiss.write_index(index, "documents/index.faiss")