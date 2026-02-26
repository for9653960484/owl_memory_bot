"""
Telegram-бот с короткой и длинной памятью.
• Короткая память: последние 10 сообщений диалога (history buffer)
• Длинная память: документы → эмбеддинги → ChromaDB (RAG)
Загружайте PDF/TXT/DOCX, общайтесь — бот помнит и документы, и контекст разговора.
"""

import asyncio
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from docx import Document as DocxDocument
from openai import AsyncOpenAI
from pypdf import PdfReader

import chromadb
from chromadb.utils import embedding_functions

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
CHROMA_PATH = "./memory"

# Короткая память: сколько последних сообщений хранить
HISTORY_SIZE = 10

# Длинная память: разбиение документа
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_TOP_K = 5

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Клиенты
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY",
    model_name="text-embedding-3-small",
)

dp = Dispatcher()

# Короткая память: user_id -> deque последних N сообщений
user_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))


# =============================================================================
# КОРОТКАЯ ПАМЯТЬ (history buffer)
# =============================================================================


def get_history_for_api(user_id: int) -> list[dict]:
    """История диалога в формате OpenAI API."""
    return [{"role": m["role"], "content": m["content"]} for m in user_history[user_id]]


def add_to_history(user_id: int, role: str, content: str) -> None:
    """Добавить сообщение в короткую память."""
    user_history[user_id].append({"role": role, "content": content})


# =============================================================================
# ДЛИННАЯ ПАМЯТЬ (документ → ChromaDB)
# =============================================================================


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Разбивает текст на части с перекрытием."""
    if not text or not text.strip():
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if overlap < chunk_size else end
    return chunks


def load_document(file_path: str) -> str:
    """Извлекает текст из PDF, TXT, DOCX."""
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            parts = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(p for p in parts if p.strip())
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return ""
    except Exception:
        return ""


def embed_chunks(user_id: int, chunks: list[str], doc_name: str = "") -> None:
    """Сохраняет chunks в ChromaDB с эмбеддингами."""
    if not chunks:
        return

    collection_name = f"user_{user_id}"
    try:
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=openai_ef
        )
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"},
        )

    existing = collection.count()
    ids = [f"{doc_name}_{i}_{existing}" for i in range(len(chunks))]
    metadatas = [{"doc": doc_name, "chunk_idx": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_context(user_id: int, query: str, top_k: int = RETRIEVE_TOP_K) -> str:
    """Поиск релевантных фрагментов в ChromaDB по запросу."""
    collection_name = f"user_{user_id}"
    try:
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=openai_ef
        )
    except Exception:
        return ""

    if collection.count() == 0:
        return ""

    results = collection.query(
        query_texts=[query], n_results=min(top_k, collection.count())
    )
    if not results or not results["documents"] or not results["documents"][0]:
        return ""
    return "\n\n---\n\n".join(results["documents"][0])


# =============================================================================
# ГЕНЕРАЦИЯ ОТВЕТА (объединяет обе памяти)
# =============================================================================


async def generate_reply(user_id: int, user_message: str) -> str:
    """
    Формирует ответ, используя короткую и длинную память.
    1. Ищет релевантные фрагменты документов (длинная память)
    2. Собирает промпт: контекст документов + история диалога + текущее сообщение
    3. Отправляет в OpenAI
    """
    # Длинная память: релевантные фрагменты по запросу
    doc_context = retrieve_context(user_id, user_message)

    # Короткая память: история диалога
    history = get_history_for_api(user_id)

    # Системный промпт: объединяем инструкции и контекст документов (если есть)
    if doc_context.strip():
        system = f"""Ты полезный помощник. У тебя два источника:
1. Контекст из загруженных документов — отвечай по нему строго, не выдумывай. Если ответа нет в контексте — скажи об этом.
2. История диалога — учитывай для связности.

[Контекст из документов]
{doc_context}
[/Контекст]"""
    else:
        system = "Ты полезный помощник. Учитывай историю диалога для связности разговора."

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


# =============================================================================
# ОБРАБОТЧИКИ TELEGRAM
# =============================================================================


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    """Загрузка документа → текст → chunks → эмбеддинги → ChromaDB."""
    user_id = message.from_user.id if message.from_user else 0
    doc = message.document

    if not doc.file_name:
        await message.answer("Не удалось определить имя файла.")
        return

    ext = Path(doc.file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await message.answer(
            f"Поддерживаются только: PDF, TXT, DOCX. Получен: {ext or '?'}"
        )
        return

    await message.answer("Обрабатываю документ...")

    try:
        file = await message.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            await message.bot.download_file(file.file_path, tmp.name)
            tmp_path = tmp.name

        try:
            text = load_document(tmp_path)
            if not text or len(text.strip()) < 10:
                await message.answer(
                    "Не удалось извлечь текст (пусто или ошибка)."
                )
                return

            chunks = _chunk_text(text)
            if not chunks:
                await message.answer("Текст слишком короткий.")
                return

            doc_name = Path(doc.file_name).stem
            embed_chunks(user_id, chunks, doc_name=doc_name)

            await message.answer(
                f"Документ «{doc.file_name}» обработан. Добавлено фрагментов: {len(chunks)}.\n"
                f"Можете задавать вопросы или просто общаться."
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    """Текстовое сообщение → короткая + длинная память → ответ."""
    user_id = message.from_user.id if message.from_user else 0
    user_text = (message.text or "").strip()

    if not user_text:
        await message.answer("Отправьте текст или загрузите документ (PDF, TXT, DOCX).")
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        reply = await generate_reply(user_id, user_text)

        # Добавляем в короткую память
        add_to_history(user_id, "user", user_text)
        add_to_history(user_id, "assistant", reply)

        await message.answer(reply)

    except Exception as e:
        await message.answer(f"Ошибка: {e}")


async def main() -> None:
    """Запуск бота."""
    if not BOT_TOKEN:
        raise ValueError("Задайте BOT_TOKEN в .env или переменных окружения")
    if not OPENAI_API_KEY:
        raise ValueError("Задайте OPENAI_API_KEY в .env или переменных окружения")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    bot = Bot(token=BOT_TOKEN)
    print("Бот запущен. Короткая + длинная память активны.")
    print("Загружайте документы и общайтесь.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
