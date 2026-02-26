"""
Telegram-бот с короткой памятью (history buffer).
Хранит последние 10 сообщений диалога для каждого пользователя.
Использует aiogram 3.x и OpenAI ChatCompletion API.
"""

import asyncio
import os
from collections import defaultdict
from collections import deque

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from openai import AsyncOpenAI

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

# Количество последних сообщений диалога на одного пользователя
HISTORY_SIZE = 10

# Модель OpenAI (gpt-4, gpt-4-turbo, gpt-3.5-turbo и т.д.)
OPENAI_MODEL = "gpt-3.5-turbo"

# Токены из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Клиент OpenAI
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Диспетчер aiogram 3.x
dp = Dispatcher()

# Память: user_id -> deque из последних N сообщений {"role": ..., "content": ...}
# deque с maxlen автоматически удаляет старые сообщения
user_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))


def get_history_for_api(user_id: int) -> list[dict]:
    """
    Возвращает историю диалога в формате OpenAI API.
    Каждый элемент: {"role": "user" | "assistant", "content": "..."}
    """
    history = user_history[user_id]
    return [{"role": msg["role"], "content": msg["content"]} for msg in history]


async def get_ai_response(user_id: int, user_message: str) -> str:
    """
    Отправляет в OpenAI историю диалога + текущее сообщение и получает ответ.
    """
    history = get_history_for_api(user_id)
    messages = history + [{"role": "user", "content": user_message}]

    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


@dp.message()
async def handle_message(message: Message) -> None:
    """
    Обработчик входящих сообщений.
    1. Запрашивает ответ у OpenAI (история + текущее сообщение)
    2. Добавляет в историю сообщение пользователя и ответ бота
    3. Отправляет ответ пользователю
    """
    user_id = message.from_user.id if message.from_user else 0
    user_text = message.text or ""

    if not user_text.strip():
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return

    # Показываем, что бот печатает
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Запрос к OpenAI: история диалога + текущее сообщение
        ai_reply = await get_ai_response(user_id, user_text)

        # Добавляем в историю: сообщение пользователя и ответ бота
        user_history[user_id].append({"role": "user", "content": user_text})
        user_history[user_id].append({"role": "assistant", "content": ai_reply})

        await message.answer(ai_reply)

    except Exception as e:
        await message.answer(f"Ошибка при обработке: {e}")


async def main() -> None:
    """Запуск бота."""
    if not BOT_TOKEN:
        raise ValueError("Переменная окружения BOT_TOKEN не задана")
    if not OPENAI_API_KEY:
        raise ValueError("Переменная окружения OPENAI_API_KEY не задана")

    bot = Bot(token=BOT_TOKEN)
    print("Бот запущен. Ожидание сообщений...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
