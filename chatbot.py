import os
import sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import time
import traceback
import pickle
import json
import pandas as pd  # اضافه شد برای خواندن CSV
import requests  # اضافه شد برای دریافت داده از اینترنت

# 1. تنظیم توکن Hugging Face API
# در Google Colab، بهتر است توکن را به عنوان یک متغیر محیطی تنظیم کنید
# در Colab: from google.colab import userdata
# TOKEN = userdata.get('HF_TOKEN')
# os.environ["HF_TOKEN"] = TOKEN



# مسیر فایل‌های کش
CACHE_DIR = "cache"
PDF_STATE_PATH = os.path.join(CACHE_DIR, "pdf_state.json")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMBEDDING_MODEL_PATH = os.path.join(CACHE_DIR, "embedding_model_name.txt")
USER_HISTORY_PATH = os.path.join(CACHE_DIR, "user_history.json")

HISTORY_TURNS = 5  # تعداد پیام‌های قبلی که به مدل می‌دهیم

os.makedirs(CACHE_DIR, exist_ok=True)

# 2. تابع خواندن PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        pass
    return text

# تابع جدید برای خواندن و تبدیل CSV به لیست کالاها
# هر ردیف یک chunk است

def extract_rows_from_csv(csv_path):
    if csv_path.startswith('http://') or csv_path.startswith('https://'):
        response = requests.get(csv_path)
        response.encoding = 'utf-8'
        df = pd.read_csv(io.StringIO(response.text))
    else:
        df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        # فرض: ستون‌های name, description, price, stock
        text = f"نام: {row['name']}\nتوضیحات: {row['description']}\nقیمت: {row['price']}\nموجودی: {row['stock']}"
        rows.append(text)
    return rows

# 3. تقسیم‌بندی متن (Text Chunking)
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# 4. مدل جاسازی (Embedding Model) و ساخت FAISS Index
def create_faiss_index(chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(embedding_model_name)
    if not chunks:
        # اگر chunks خالی بود، یک index خالی با dimension مناسب ایجاد کن
        dummy_embedding = embedding_model.encode(["dummy text"])
        dimension = dummy_embedding.shape[1]
        index = faiss.IndexFlatL2(dimension)
        return index, embedding_model
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embedding_model

# 5. بازیابی (Retrieval)
def retrieve_relevant_chunks(query, index, embedding_model, chunks, k=5):
    if not chunks:
        return ""
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return " ".join(relevant_chunks)

# 6. مدل پرسش و پاسخ (Question Answering Model)
def load_qa_model(model_name="marzinouri/parsbert-finetuned-persianQA"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return qa_pipeline
    except Exception as e:
        device = 0 if torch.cuda.is_available() else -1
        try:
            qa_pipeline = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device=device)
            return qa_pipeline
        except Exception as gen_e:
            raise


# تابع جدید برای تولید متن با مدل لوکال فارسی (gpt2-fa)
def get_local_completion(prompt, model_pipeline, max_tokens=30):
    try:
        # mbart نیاز به تعیین زبان ورودی و خروجی دارد
        forced_bos_token_id = model_pipeline.tokenizer.lang_code_to_id.get("fa_IR", None)
        output = model_pipeline(
            prompt,
            max_new_tokens=max(60, max_tokens),
            forced_bos_token_id=forced_bos_token_id
        )
        print("[DEBUG] Raw model output:", output)
        if isinstance(output, list):
            if "generated_text" in output[0]:
                return output[0]["generated_text"]
            elif "text" in output[0]:
                return output[0]["text"]
            else:
                return str(output[0])
        return str(output)
    except Exception as e:
        print("[ERROR] get_local_completion:", e)
        return "متاسفانه در حال حاضر قادر به پاسخگویی نیستم. لطفاً بعداً امتحان کنید."

# 7. تابع چت
def summarize_context(context, max_length=2000, head=1000, tail=1000):
    if len(context) <= max_length:
        return context
    # Extractive: keep start and end
    return context[:head] + '\n...\n' + context[-tail:]

def chat_with_model(index, embedding_model, chunks, model_pipeline):
    history = load_user_history()
    GREEN = '\033[92m'
    RESET = '\033[0m'
    while True:
        user_query = input(f"{GREEN}شما: {RESET}")
        if user_query.lower() == 'خروج':
            break
    # فقط آخرین پیام کاربر به مدل داده شود
    history.append({"role": "user", "content": user_query})
    prompt_for_llm = user_query.strip()
    answer = get_local_completion(prompt_for_llm, model_pipeline, max_tokens=120)
    print(answer)
    # ذخیره پاسخ در تاریخچه
    history.append({"role": "assistant", "content": answer})
    save_user_history(history)

def get_pdf_state(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    state = {}
    for f in pdf_files:
        path = os.path.join(pdf_folder, f)
        try:
            mtime = os.path.getmtime(path)
            state[f] = mtime
        except Exception:
            continue
    return state

def load_cached_data():
    try:
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        index = faiss.read_index(INDEX_PATH)
        with open(EMBEDDING_MODEL_PATH, 'r', encoding='utf-8') as f:
            embedding_model_name = f.read().strip()
        embedding_model = SentenceTransformer(embedding_model_name)
        return chunks, index, embedding_model
    except Exception:
        return None, None, None

def save_cached_data(chunks, index, embedding_model_name):
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_PATH)
    with open(EMBEDDING_MODEL_PATH, 'w', encoding='utf-8') as f:
        f.write(embedding_model_name)

def load_pdf_state():
    if not os.path.exists(PDF_STATE_PATH):
        return None
    try:
        with open(PDF_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def save_pdf_state(state):
    with open(PDF_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f)

def load_user_history():
    if not os.path.exists(USER_HISTORY_PATH):
        return []
    try:
        with open(USER_HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_user_history(history):
    with open(USER_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)

# Main execution logic
if __name__ == "__main__":
    csv_path = "https://docs.google.com/spreadsheets/d/1n5kXGOwOzVHmmdNIkhzJUaNaf_6qgJNMyz79gp5jl8E/export?format=csv"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    force_rebuild = False
    user_input = input("آیا می‌خواهید کش را بازسازی کنید؟ (y/n): ").strip().lower()
    if user_input == 'y':
        force_rebuild = True
    # بررسی تغییرات فایل CSV
    def is_remote_url(path):
        return path.startswith('http://') or path.startswith('https://')

    if is_remote_url(csv_path):
        try:
            resp = requests.head(csv_path, allow_redirects=True, timeout=10)
            csv_exists = resp.status_code == 200
        except Exception:
            csv_exists = False
        csv_mtime = None  # نمی‌توان mtime را برای URL گرفت
    else:
        csv_exists = os.path.exists(csv_path)
        csv_mtime = os.path.getmtime(csv_path) if csv_exists else None

    state_path = os.path.join(CACHE_DIR, "csv_state.json")
    cached_state = None
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            cached_state = json.load(f)
    need_rebuild = (not cached_state or cached_state.get('mtime') != csv_mtime or force_rebuild)
    if not need_rebuild:
        chunks, index, embedding_model = load_cached_data()
        if chunks is None or index is None or embedding_model is None:
            need_rebuild = True
    if need_rebuild:
        if not csv_exists:
            print(f"فایل {csv_path} پیدا نشد یا قابل دسترسی نیست!")
            exit(1)
        print(f"در حال خواندن فایل: {csv_path}")
        rows = extract_rows_from_csv(csv_path)
        print(f"✅ تعداد کالاها: {len(rows)}")
        chunks = rows
        index, embedding_model = create_faiss_index(chunks, embedding_model_name=embedding_model_name)
        save_cached_data(chunks, index, embedding_model_name)
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump({'mtime': csv_mtime}, f)
    # بارگذاری مدل لوکال فارسی سبک (gpt2-fa)
    print("در حال بارگذاری مدل فارسی لوکال (gpt2-fa)...")
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
    model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa")
    device = 0 if torch.cuda.is_available() else -1
    model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    print("مدل فارسی آماده است.")
    chat_with_model(index, embedding_model, chunks, model_pipeline)