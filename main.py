from transformers import pipeline, AutoTokenizer

model_name = "facebook/opt-125m"
chatbot = pipeline("text-generation", model=model_name, device=0)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}"

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)


response = chatbot(formatted_input, max_length=100, truncation=True)
print(response[0]['generated_text'])