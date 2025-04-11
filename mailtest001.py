import imaplib
import email
from email.header import decode_header
import time
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_checker.log'),
        logging.StreamHandler()
    ]
)

def get_email_credentials():
    """Получаем учетные данные из файла или переменных окружения"""
    # В реальной программе лучше использовать безопасное хранение паролей
    # Например, через configparser или переменные окружения
    return {
        'email': 'mephi1984@yandex.ru',
        'password': '',
        'imap_server': 'imap.yandex.ru'
    }

def connect_to_imap(imap_server, email, password):
    """Устанавливаем соединение с IMAP сервером"""
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email, password)
        mail.select('INBOX')  # Выбираем папку входящих
        return mail
    except Exception as e:
        logging.error(f"Ошибка подключения к IMAP: {e}")
        return None

def decode_mime_words(s):
    """Декодируем строку с MIME-кодировкой"""
    return ''.join(
        word.decode(encoding or 'utf-8') if isinstance(word, bytes) else word
        for word, encoding in decode_header(s)
    )

def process_new_emails(mail, last_checked_uid):
    """Обрабатываем новые письма, игнорируя старые"""
    cutoff_date = "10-Apr-2025"  # Фильтруем письма старее этой даты

    try:
        # Комбинированный поиск: новые UID И дата после cutoff_date
        result, data = mail.uid('search', None,
                                f'(UID {last_checked_uid + 1}:*) (SINCE "{cutoff_date}")')
        if result != 'OK':
            logging.error("Ошибка при поиске писем")
            return last_checked_uid

        email_uids = data[0].split()
        if not email_uids:
            return last_checked_uid

        max_uid = last_checked_uid
        for email_uid in email_uids:
            email_uid = int(email_uid)
            max_uid = max(max_uid, email_uid)

            result, data = mail.uid('fetch', str(email_uid), '(RFC822)')
            if result != 'OK':
                continue

            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = decode_mime_words(msg['Subject'])
            from_ = decode_mime_words(msg['From'])
            date = msg['Date']

            logging.info(f"Новое письмо! UID: {email_uid}")
            logging.info(f"От: {from_}")
            logging.info(f"Тема: {subject}")
            logging.info(f"Дата: {date}")
            logging.info("-" * 50)

        return max_uid
    except Exception as e:
        logging.error(f"Ошибка при обработке писем: {e}")
        return last_checked_uid


def main():
    credentials = get_email_credentials()
    last_checked_uid = 0  # Начинаем с 0, чтобы получить все письма при первом запуске

    while True:
        try:
            logging.info(f"Проверка новых писем в {datetime.now()}")

            mail = connect_to_imap(
                credentials['imap_server'],
                credentials['email'],
                credentials['password']
            )

            if mail:
                last_checked_uid = process_new_emails(mail, last_checked_uid)
                mail.logout()

            # Пауза между проверками (например, 1 минута)
            time.sleep(60)

        except KeyboardInterrupt:
            logging.info("Программа остановлена пользователем")
            break
        except Exception as e:
            logging.error(f"Неожиданная ошибка: {e}")
            time.sleep(300)  # Ждем 5 минут перед повторной попыткой


if __name__ == "__main__":
    main()