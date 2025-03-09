from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset

# Загрузите датасет wiki_dpr
wiki_dpr_dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)

# Инициализируйте токенизатор и ретривер
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="exact",
    use_dummy_dataset=False,  # Используйте реальный датасет
    dataset=wiki_dpr_dataset  # Передайте загруженный датасет
)

# Инициализируйте модель
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Подготовьте входные данные
input_dict = tokenizer.prepare_seq2seq_batch("how many countries are in europe", return_tensors="pt")

# Сгенерируйте ответ
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])