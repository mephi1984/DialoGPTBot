from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from telegram import ReactionTypeEmoji
import asyncio
import random

TOKEN = ''

BOT_USERNAME = "@FishRunGamesBot"

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    chat_history_ids = None
    await update.message.reply_text('Привет! Я ваш телеграм-бот с DialoGPT. Начнем диалог!')


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    print("chat update")

    if update.message.chat.type in ["group", "supergroup"]:
        if not update.message.text:
            return
        user_message = update.message.text.replace(BOT_USERNAME, "").strip()
    else:
        user_message = update.message.text

    # Проверяем, обращено ли сообщение к боту (для групп)
    is_directed_to_bot = (
            update.message.chat.type in ["private"] or
            (update.message.chat.type in ["group", "supergroup"] and
             BOT_USERNAME.lower() in update.message.text.lower())
    )

    # Проверяем на наличие ключевых фраз для реакций
    if "good luck" in user_message.lower():
        await asyncio.sleep(random.uniform(2, 5))
        await update.message.set_reaction([ReactionTypeEmoji('👍')], is_big=False)
    elif "love" in user_message.lower():
        await asyncio.sleep(random.uniform(2, 5))
        await update.message.set_reaction([ReactionTypeEmoji('❤️')], is_big=False)


    # Для сообщений не к боту - 30% вероятность ответа
    if not is_directed_to_bot and random.random() > 0.3:
        return

    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                              dim=-1) if chat_history_ids is not None else new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Добавляем случайный эмодзи с вероятностью 70%
    if random.random() < 0.7:
        emoji = random.choice(['👀', '🤔', '😂'])
        bot_response = f"{bot_response} {emoji}"

    await update.message.chat.send_chat_action("typing")
    await asyncio.sleep(random.uniform(2, 5))
    await update.message.reply_text(bot_response)


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling()

if __name__ == '__main__':
    main()