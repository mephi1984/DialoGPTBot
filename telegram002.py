from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Токен вашего бота
TOKEN = ''


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


chat_history_ids = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    chat_history_ids = None
    await update.message.reply_text('Hello, I am a DialoGPT model!')


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    user_message = update.message.text
    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    await update.message.reply_text(bot_response)


def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    application.run_polling()

if __name__ == '__main__':
    main()