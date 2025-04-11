from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from telegram import ReactionTypeEmoji
import asyncio
from queue import Queue
import random

from concurrent.futures import ThreadPoolExecutor

import imaplib
import email
from email.header import decode_header
import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

IMAP_CONFIG = {
        'email': 'mephi1984@yandex.ru',
        'password': '',
        'imap_server': 'imap.yandex.ru'
    }

TOKEN = ''

BOT_USERNAME = "@FishRunGamesBot"

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

ADMIN_CHAT_ID = -1002520236252

chat_history_ids = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    chat_history_ids = None
    await update.message.reply_text('Привет! Я ваш телеграм-бот с DialoGPT. Начнем диалог!')


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_history_ids
    print("chat update")
    # -1002520236252
    print(update.message.chat.id)

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


async def check_and_notify(context):
    """Асинхронная проверка почты с таймаутом"""
    try:
        # Ограничиваем время выполнения (например, 4 минуты)
        async with asyncio.timeout(240):
            new_emails = await asyncio.to_thread(check_new_emails)

            for email in new_emails:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=f"📧You have a new email, boss!\n**From:** {email['from']}\n**Subject:** {email['subject']}"
                )
    except asyncio.TimeoutError:
        logging.warning("Проверка почты превысила лимит времени")
    except Exception as e:
        logging.error(f"Ошибка: {e}")



def check_new_emails():
    """
    Синхронная функция проверки почты с фильтрацией по дате
    Возвращает список словарей с информацией о новых письмах
    Формат: [{'from': str, 'subject': str, 'date': datetime}, ...]
    """
    cutoff_date = datetime(2025, 4, 10, tzinfo=timezone.utc)
    new_emails = []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_CONFIG['imap_server'])
        mail.login(IMAP_CONFIG['email'], IMAP_CONFIG['password'])
        mail.select('INBOX')

        # Ищем непрочитанные письма НОВЕЕ 10 апреля 2025
        status, messages = mail.search(None, 'UNSEEN', 'SINCE "10-Apr-2025"')

        if status == 'OK' and messages[0]:
            for num in messages[0].split():
                status, data = mail.fetch(num, '(RFC822)')
                if status == 'OK':
                    msg = email.message_from_bytes(data[0][1])

                    # Обработка даты письма
                    email_date = parsedate_to_datetime(msg['Date'])
                    if email_date.tzinfo is None:
                        email_date = email_date.replace(tzinfo=timezone.utc)

                    if email_date < cutoff_date:
                        continue

                    # Декодирование темы
                    subject = decode_header(msg['Subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode('utf-8', errors='ignore')

                    # Декодирование отправителя
                    from_ = decode_header(msg['From'])[0][0]
                    if isinstance(from_, bytes):
                        from_ = from_.decode('utf-8', errors='ignore')

                    # Добавляем письмо в результат
                    new_emails.append({
                        'from': from_,
                        'subject': subject,
                        'date': email_date,
                        'raw_message': msg  # Опционально, если нужен доступ к телу письма
                    })

        mail.logout()
    except Exception as e:
        logging.error(f"Ошибка при проверке почты: {e}")
        # В случае ошибки возвращаем пустой список
        return []

    return new_emails



async def process_bot_queue(context):
    """Обработчик очереди сообщений"""
    while not context.application.bot_queue.empty():
        message = context.application.bot_queue.get()
        await context.bot.send_message(
            chat_id=message['chat_id'],
            text=message['text']
        )

def main():
    application = Application.builder().token(TOKEN).build()
    application.bot_queue = Queue()

    application.job_queue.scheduler.configure(max_instances=1)

    # Запускаем проверку каждые 5 минут
    application.job_queue.run_repeating(
        check_and_notify,
        interval=300,
        first=10,
        job_kwargs={'misfire_grace_time': 60}  # Допустимая задержка
    )

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    application.run_polling()

if __name__ == '__main__':
    main()

