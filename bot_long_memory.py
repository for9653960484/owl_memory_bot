"""
Telegram-бот с долгой памятью (документ → эмбеддинги → ChromaDB).
Загружайте PDF/TXT/DOCX, задавайте вопросы — бот отвечает на основе содержимого документа.
Использует aiogram 3.x, OpenAI Embeddings + ChatCompletion, ChromaDB.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from docx import Document as DocxDocument
from openai import AsyncOpenAI
from pypdf import PdfReader

import chromadb
from chromadb.utils import embedding_functions

# python-dotenv для загрузки .env (BOT_TOKEN, OPENAI_API_KEY)
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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_TOP_K = 5

# Поддерживаемые расширения
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Клиенты
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY",
    model_name="text-embedding-3-small",
)

dp = Dispatcher()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Разбивает текст на части (chunks) с перекрытием.
    overlap — сколько символов пересекаются между соседними чанками.
    """
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
    """
    Извлекает текст из документа (PDF, TXT, DOCX).
    Возвращает полный текст или пустую строку при ошибке.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n\n".join(parts)

        elif ext == ".docx":
            doc = DocxDocument(file_path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        else:
            return ""
    except Exception:
        return ""


def embed_chunks(user_id: int, chunks: list[str], doc_name: str = "") -> None:
    """
    Конвертирует chunks в эмбеддинги и сохраняет в ChromaDB.
    Для каждого user_id — своя коллекция.
    """
    if not chunks:
        return

    collection_name = f"user_{user_id}"
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=openai_ef,
        )
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ID для каждого чанка (уникальные в рамках коллекции)
    existing = collection.count()
    ids = [f"{doc_name}_{i}_{existing}" for i in range(len(chunks))]
    metadatas = [{"doc": doc_name, "chunk_idx": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def retrieve_context(user_id: int, query: str, top_k: int = RETRIEVE_TOP_K) -> str:
    """
    Поиск релевантных фрагментов в векторной базе по вопросу.
    Возвращает объединённый контекст для передачи модели.
    """
    collection_name = f"user_{user_id}"
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=openai_ef,
        )
    except Exception:
        return ""

    if collection.count() == 0:
        return ""

    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    if not results or not results["documents"] or not results["documents"][0]:
        return ""

    return "\n\n---\n\n".join(results["documents"][0])


async def answer_question(context: str, question: str) -> str:
    """
    Отправляет контекст + вопрос в OpenAI и получает ответ.
    Модель инструктирована отвечать только на основе контекста.
    """
    if not context.strip():
        return "Сначала загрузите документ (PDF, TXT или DOCX)."

    system = """Ты отвечаешь строго на основе предоставленного контекста из документа.
Не выдумывай информацию. Если ответа нет в контексте — так и скажи."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Контекст из документа:\n\n{context}\n\nВопрос: {question}"},
    ]

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
    """
    Пользователь загружает файл → сохраняем, извлекаем текст, chunk'им, эмбедим, сохраняем в ChromaDB.
    """
    user_id = message.from_user.id if message.from_user else 0
    doc = message.document

    if not doc.file_name:
        await message.answer("Не удалось определить имя файла.")
        return

    ext = Path(doc.file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await message.answer(
            f"Поддерживаются только: PDF, TXT, DOCX.\n"
            f"Получен: {ext or 'неизвестный формат'}"
        )
        return

    await message.answer("Обрабатываю документ...")

    try:
        # Скачиваем файл во временную папку
        file = await message.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            await message.bot.download_file(file.file_path, tmp.name)
            tmp_path = tmp.name

        try:
            # 1. Извлекаем текст
            text = load_document(tmp_path)
            if not text or len(text.strip()) < 10:
                await message.answer("Не удалось извлечь текст из документа (пусто или ошибка).")
                return

            # 2. Разбиваем на chunks
            chunks = _chunk_text(text)
            if not chunks:
                await message.answer("Текст слишком короткий для обработки.")
                return

            # 3. Эмбедим и сохраняем в ChromaDB
            doc_name = Path(doc.file_name).stem
            embed_chunks(user_id, chunks, doc_name=doc_name)

            await message.answer(
                f"Документ «{doc.file_name}» обработан.\n"
                f"Добавлено фрагментов: {len(chunks)}.\n"
                f"Теперь можете задавать вопросы по нему."
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        await message.answer(f"Ошибка при обработке: {e}")


@dp.message(F.text)
async def handle_question(message: Message) -> None:
    """
    Текстовое сообщение считаем вопросом → поиск контекста в ChromaDB → ответ через OpenAI.
    """
    user_id = message.from_user.id if message.from_user else 0
    question = (message.text or "").strip()

    if not question:
        await message.answer("Напишите вопрос по загруженному документу.")
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # 4–5. Поиск релевантных фрагментов
        context = retrieve_context(user_id, question)

        # 6–7. Ответ на основе контекста
        answer = await answer_question(context, question)
        await message.answer(answer)

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
    print("Бот запущен. Загружайте документы и задавайте вопросы.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
