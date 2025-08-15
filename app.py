from flask import Flask, request, render_template_string, jsonify, make_response
import os
import uuid
import json
from chatbot import extract_text_from_pdf, chunk_text, create_faiss_index, retrieve_relevant_chunks, get_local_completion, load_cached_data, save_cached_data, load_pdf_state, save_pdf_state
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch
from fpdf import FPDF

PDF_FOLDER = "pdfs"
CACHE_DIR = "cache"
AVALAI_API_KEY = "aa-gYoc8tan5jfXcbWqagG9US0y7qtDa4hQzmSz1RkT9J6HewgT"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HISTORY_TURNS = 10  # تعداد پیام‌های قبلی که به مدل می‌دهیم (افزایش یافته)

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)


# بارگذاری مدل لوکال فارسی (gpt2-fa)
print("در حال بارگذاری مدل multilingual (facebook/mbart-large-50-many-to-many-mmt) ...")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
device = 0 if torch.cuda.is_available() else -1
model_pipeline = pipeline(
  "text2text-generation",
  model=model,
  tokenizer=tokenizer,
  device=device,
  max_new_tokens=128
)
print("مدل mbart-large-50 آماده است.")

# بارگذاری یا ساخت ایندکس و مدل
chunks, index, embedding_model = load_cached_data()
if chunks is None or index is None or embedding_model is None:
  pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
  if pdf_files:
    all_extracted_text = ""
    for pdf_file in pdf_files:
      pdf_path = os.path.join(PDF_FOLDER, pdf_file)
      extracted_text = extract_text_from_pdf(pdf_path)
      all_extracted_text += extracted_text + "\n"
    chunks = chunk_text(all_extracted_text)
    index, embedding_model = create_faiss_index(chunks, embedding_model_name=EMBEDDING_MODEL_NAME)
    save_cached_data(chunks, index, EMBEDDING_MODEL_NAME)
    save_pdf_state({f: os.path.getmtime(os.path.join(PDF_FOLDER, f)) for f in pdf_files})
  else:
    chunks = []
    index, embedding_model = create_faiss_index([], embedding_model_name=EMBEDDING_MODEL_NAME)

# مسیرهای جدید بر اساس session_id

def get_user_pdf_folder(session_id):
    folder = os.path.join('pdfs', session_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def get_history_path(session_id):
    folder = os.path.join('cache', session_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, 'history.json')

# تابع بارگذاری و ذخیره تاریخچه هر جلسه

def get_session_id():
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

def load_user_history(session_id):
    path = get_history_path(session_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_user_history(session_id, history):
    path = get_history_path(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)

HTML = '''
<!DOCTYPE html>
<html lang="fa">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>نمایندگی بیمه هوشمند</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Vazirmatn', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body {
      font-family: 'Vazirmatn', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      direction: rtl;
      background: #f0f2f5;
      margin: 0;
      min-height: 100vh;
      padding: 0;
      font-weight: 400;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    #main-container {
      max-width: 100%;
      margin: 0;
      background: #fff;
      border-radius: 0;
      box-shadow: none;
      padding: 0;
      border: none;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    @media (min-width: 768px) {
      #main-container {
        max-width: 900px;
        margin: 20px auto;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e4e6ea;
        min-height: calc(100vh - 40px);
        overflow: hidden;
      }
      body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
      }
    }
    #shop-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 0;
      border-bottom: 1px solid #e4e6ea;
      padding: 16px 20px;
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    #shop-logo {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      color: #fff;
      box-shadow: 0 2px 8px rgba(255,193,7,0.3);
    }
    #shop-title {
      font-size: 18px;
      font-weight: 700;
      color: #1c1e21;
      letter-spacing: -0.2px;
      font-family: 'Vazirmatn', sans-serif;
    }
    #shop-subtitle {
      font-size: 13px;
      font-weight: 400;
      color: #65676b;
      margin-top: 2px;
      font-family: 'Vazirmatn', sans-serif;
    }
    #instructions {
      background: #f8f9fa;
      padding: 16px 20px;
      border-radius: 0;
      margin: 0;
      font-size: 14px;
      color: #1c1e21;
      border-bottom: 1px solid #e4e6ea;
      line-height: 1.6;
      font-family: 'Vazirmatn', sans-serif;
      font-weight: 400;
    }
    h2 {
      margin: 0;
      color: #1c1e21;
      font-size: 16px;
      font-weight: 700;
      padding: 16px 20px 8px 20px;
      font-family: 'Vazirmatn', sans-serif;
      letter-spacing: -0.3px;
    }
    #uploadForm {
      display: flex;
      gap: 8px;
      align-items: center;
      margin: 0 20px 16px 20px;
      padding: 12px;
      background: #f8f9fa;
      border-radius: 12px;
      border: 1px solid #e4e6ea;
    }
    #uploadForm input[type="file"] {
      flex: 1;
      font-size: 14px;
      border: none;
      background: transparent;
    }
    #uploadForm button {
      background: #ff9800;
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-weight: 600;
      font-family: 'Vazirmatn', sans-serif;
      letter-spacing: -0.1px;
    }
    #uploadForm button:hover {
      background: #f57c00;
    }
    #progressBarContainer {
      width: calc(100% - 40px);
      background: #e4e6ea;
      border-radius: 8px;
      margin: 0 20px 16px 20px;
      display: none;
      overflow: hidden;
    }
    #progressBar {
      width: 0%;
      height: 16px;
      background: #4caf50;
      border-radius: 8px;
      text-align: center;
      color: white;
      font-size: 12px;
      transition: width 0.3s ease;
      line-height: 16px;
    }
    #processingMsg {
      display:none;
      color:#ff9800;
      font-weight:500;
      margin: 0 20px 8px 20px;
      font-size: 14px;
    }
    #uploadMsg {
      margin: 0 20px 8px 20px;
      color: #d32f2f;
      font-size: 14px;
    }
    #readyMsg {
      color: #2e7d32;
      font-weight: 500;
      margin: 0 20px 16px 20px;
      display: none;
      font-size: 14px;
    }
    #chatBox {
      background: #f0f2f5;
      border-radius: 0;
      min-height: 300px;
      max-height: none;
      flex: 1;
      overflow-y: auto;
      padding: 16px 12px;
      margin: 0;
      box-shadow: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
      border: none;
    }
    .bubble {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      margin-bottom: 8px;
      max-width: 85%;
    }
    .user-bubble {
      align-self: flex-end;
      flex-direction: row-reverse;
      margin-left: auto;
    }
    .bot-bubble {
      align-self: flex-start;
      margin-right: auto;
    }
    .bubble-content {
      max-width: 100%;
      padding: 12px 16px;
      border-radius: 18px;
      font-size: 15px;
      line-height: 1.6;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
      word-break: break-word;
      font-family: 'Vazirmatn', sans-serif;
      font-weight: 400;
      letter-spacing: -0.1px;
    }
    .user-bubble .bubble-content {
      background: #ff9800;
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .bot-bubble .bubble-content {
      background: #fff;
      color: #1c1e21;
      border-bottom-left-radius: 4px;
      border: none;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bubble-avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: #e4e6ea;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      font-weight: 600;
      color: #65676b;
      flex-shrink: 0;
      font-family: 'Vazirmatn', sans-serif;
    }
    .user-bubble .bubble-avatar {
      background: #ff9800;
      color: #fff;
    }
    #chatInputBar {
      display: flex;
      gap: 8px;
      align-items: center;
      margin: 0;
      padding: 16px 20px;
      background: #fff;
      border-top: 1px solid #e4e6ea;
    }
    #userInput {
      flex: 1;
      border: 1px solid #e4e6ea;
      border-radius: 20px;
      padding: 12px 16px;
      font-size: 15px;
      outline: none;
      transition: all 0.2s ease;
      background: #f0f2f5;
      font-family: 'Vazirmatn', sans-serif;
      font-weight: 400;
      letter-spacing: -0.1px;
    }
    #userInput:focus {
      border: 1px solid #ff9800;
      background: #fff;
      box-shadow: 0 0 0 2px rgba(255,152,0,0.1);
    }
    #sendBtn {
      background: #ff9800;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      font-size: 16px;
      cursor: pointer;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
      font-family: 'Vazirmatn', sans-serif;
    }
    #sendBtn:hover {
      background: #f57c00;
    }
    #shop-footer {
      margin: 0;
      padding: 12px 20px;
      border-top: 1px solid #e4e6ea;
      color: #65676b;
      font-size: 12px;
      text-align: center;
      background: #f8f9fa;
      font-family: 'Vazirmatn', sans-serif;
      font-weight: 400;
      line-height: 1.4;
    }
    .services-highlight {
      background: #f8f9fa;
      padding: 16px 20px;
      border-radius: 0;
      margin: 0;
      border-bottom: 1px solid #e4e6ea;
    }
    .services-highlight h4 {
      color: #1c1e21;
      margin: 0 0 8px 0;
      font-size: 15px;
      font-weight: 700;
      font-family: 'Vazirmatn', sans-serif;
      letter-spacing: -0.2px;
    }
    .services-highlight ul {
      margin: 0;
      padding-right: 16px;
      color: #65676b;
    }
    .services-highlight li {
      margin-bottom: 6px;
      font-size: 13px;
      line-height: 1.5;
      font-family: 'Vazirmatn', sans-serif;
      font-weight: 400;
    }
    @media (max-width: 700px) {
      #main-container { max-width: 100vw; padding: 0; }
      #chatBox { min-height: 250px; }
      #uploadForm { flex-direction: column; gap: 8px; }
      #uploadForm button { width: 100%; padding: 10px 0; font-size: 15px; }
      #userInput { font-size: 16px; padding: 12px 16px; }
      .bubble-content { font-size: 15px; padding: 12px 16px; }
      .bubble-avatar { width: 28px; height: 28px; font-size: 14px; }
      #shop-header { padding: 12px 16px; }
      #instructions, .services-highlight { padding: 12px 16px; }
      h2 { padding: 12px 16px 8px 16px; }
      #chatInputBar { padding: 12px 16px; }
      #shop-footer { padding: 8px 16px; }
    }
    @media (min-width: 768px) and (max-width: 1024px) {
      #main-container {
        max-width: 95%;
        margin: 15px auto;
        border-radius: 12px;
      }
    }
    @media (min-width: 1025px) {
      #main-container {
        max-width: 1000px;
        margin: 30px auto;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
      }
      #shop-header {
        border-radius: 20px 20px 0 0;
      }
      #chatBox {
        padding: 20px 16px;
      }
      .bubble-content {
        font-size: 16px;
        padding: 14px 18px;
      }
    }
    @media (max-width: 400px) {
      #main-container { padding: 0; }
      #chatBox { padding: 12px 8px; }
      .bubble-content { font-size: 14px; padding: 10px 14px; }
    }
  </style>
</head>
<body>
  <div id="main-container">
    <div id="shop-header">
      <div id="shop-logo">🛡️</div>
      <div>
        <div id="shop-title">نمایندگی بیمه هوشمند</div>
        <div id="shop-subtitle">اولین و تنها نمایندگی هوشمند سراسر کشور</div>
      </div>
    </div>
    <div class="services-highlight">
      <h4>🌟 ویژگی‌ها و خدمات نمایندگی بیمه هوشمند:</h4>
      <ul>
        <li>✅ اولین و تنها نمایندگی هوشمند سراسر کشور</li>
        <li>✅ مشاوره تخصصی و رایگان بیمه</li>
        <li>✅ صدور انواع بیمه‌نامه‌ها (شخص ثالث، بدنه، آتش‌سوزی، عمر و...)</li>
        <li>✅ پیگیری خسارت و پشتیبانی ۲۴ ساعته</li>
        <li>✅ استعلام و مقایسه قیمت بیمه از شرکت‌های مختلف</li>
      </ul>
    </div>
    <div id="instructions">
      <h3>🛡️ راهنمای استفاده از دستیار هوشمند بیمه:</h3>
      <ol>
        <li>ابتدا مدارک بیمه (مانند بیمه‌نامه قبلی، گزارش خسارت و...) را از طریق فرم زیر بارگذاری نمایید (اختیاری).</li>
        <li>پس از اتمام بارگذاری و پردازش، پیام "آماده چت است" نمایش داده می‌شود.</li>
        <li>سپس می‌توانید سوالات خود را درباره انواع بیمه، شرایط، استعلام قیمت و ... در بخش چت وارد کنید.</li>
        <li>دستیار هوشمند ما آماده پاسخگویی به سوالات شما و ارائه مشاوره بیمه‌ای است.</li>
      </ol>
    </div>
    <h2>📄 آپلود مدارک بیمه (PDF)</h2>
    <form id="uploadForm" enctype="multipart/form-data">
      <input type="file" name="pdf" accept=".pdf">
      <button type="submit">آپلود</button>
    </form>
    <div id="progressBarContainer">
      <div id="progressBar">0%</div>
    </div>
    <div id="processingMsg">در حال آماده‌سازی فایل...</div>
    <div id="uploadMsg"></div>
    <div id="readyMsg">آماده چت است! اکنون می‌توانید سوالات خود را بپرسید.</div>
    <hr>
    <h2>💬 گفتگو با دستیار بیمه</h2>
    <div id="chatBox"></div>
    <div id="chatInputBar">
      <input type="text" id="userInput" placeholder="سوال خود را درباره محصولات یا سفارشات بپرسید..." autocomplete="off" onkeydown="if(event.key==='Enter'){sendMessage();return false;}">
      <button id="sendBtn" onclick="sendMessage()">ارسال</button>
    </div>
    <div id="shop-footer">
      <div>🛒 فروشگاه آنلاین شاپ &bull; تلفن: 021-98765432 &bull; آدرس: تهران، خیابان انقلاب، پلاک ۲۰</div>
      <div style="margin-top:4px; color:#ff9800; font-size:13px;">تمامی حقوق محفوظ است &copy; 2025</div>
    </div>
  </div>
  <script>
    // --- آپلود PDF ---
    document.getElementById('uploadForm').onsubmit = async function(e) {
      e.preventDefault();
      let formData = new FormData(this);
      let xhr = new XMLHttpRequest();
      let progressBarContainer = document.getElementById('progressBarContainer');
      let progressBar = document.getElementById('progressBar');
      let uploadMsg = document.getElementById('uploadMsg');
      let readyMsg = document.getElementById('readyMsg');
      let processingMsg = document.getElementById('processingMsg');
      progressBarContainer.style.display = 'block';
      progressBar.style.width = '0%';
      progressBar.innerText = '0%';
      readyMsg.style.display = 'none';
      uploadMsg.innerText = '';
      processingMsg.style.display = 'none';
      xhr.upload.onprogress = function(event) {
        if (event.lengthComputable) {
          let percent = Math.round((event.loaded / event.total) * 100);
          progressBar.style.width = percent + '%';
          progressBar.innerText = percent + '%';
        }
      };
      xhr.onloadstart = function() {
        processingMsg.style.display = 'none';
      };
      xhr.onload = function() {
        progressBar.style.width = '100%';
        progressBar.innerText = '100%';
        if (xhr.status === 200) {
          uploadMsg.innerText = xhr.responseText;
          processingMsg.style.display = 'block';
          readyMsg.style.display = 'none';
          setTimeout(function() {
            processingMsg.style.display = 'none';
            readyMsg.style.display = 'block';
          }, 1200);
        } else {
          uploadMsg.innerText = xhr.responseText;
          readyMsg.style.display = 'none';
          processingMsg.style.display = 'none';
        }
        setTimeout(function() {
          progressBarContainer.style.display = 'none';
        }, 1500);
      };
      xhr.open('POST', '/upload', true);
      xhr.send(formData);
    };
    // --- چت ---
    let chatBox = document.getElementById('chatBox');
    function addMessage(msg, sender) {
      let bubble = document.createElement('div');
      bubble.className = 'bubble ' + (sender === 'user' ? 'user-bubble' : 'bot-bubble');
      let avatar = document.createElement('div');
      avatar.className = 'bubble-avatar';
      avatar.innerHTML = sender === 'user' ? '👤' : '🛒';
      let content = document.createElement('div');
      content.className = 'bubble-content';
      content.innerText = msg;
      bubble.appendChild(avatar);
      bubble.appendChild(content);
      chatBox.appendChild(bubble);
      chatBox.scrollTop = chatBox.scrollHeight;
    }
    async function sendMessage() {
      let input = document.getElementById('userInput');
      let msg = input.value.trim();
      if (!msg) return;
      addMessage(msg, 'user');
      input.value = "";
      input.focus();
      let res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      let data = await res.json();
      addMessage(data.answer, 'bot');
    }
  </script>
</body>
</html>
'''

HTML = HTML.replace(
    '<div id="chatInputBar">',
    '<a href="/download_summary" target="_blank" style="display:block;text-align:center;margin:16px 20px;">\n'
    '  <button style="background:#ff9800;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:600;transition:all 0.2s ease;font-family:\'Vazirmatn\',sans-serif;letter-spacing:-0.1px;box-shadow:0 2px 8px rgba(255,193,7,0.3);">\n'
    '    دانلود خلاصه سفارش یا فاکتور\n'
    '  </button>\n'
    '</a>\n<div id="chatInputBar">'
)

@app.route("/", methods=["GET"])
def home():
    resp = make_response(render_template_string(HTML))
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        resp.set_cookie('session_id', session_id)
    return resp

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chunks, index, embedding_model
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    user_pdf_folder = get_user_pdf_folder(session_id)
    file = request.files.get("pdf")
    if file and file.filename.endswith(".pdf"):
        save_path = os.path.join(user_pdf_folder, file.filename)
        file.save(save_path)
        # بازسازی ایندکس و مدل فقط برای این کاربر
        pdf_files = [f for f in os.listdir(user_pdf_folder) if f.lower().endswith('.pdf')]
        all_extracted_text = ""
        for pdf_file in pdf_files:
            pdf_path = os.path.join(user_pdf_folder, pdf_file)
            extracted_text = extract_text_from_pdf(pdf_path)
            all_extracted_text += extracted_text + "\n"
        global chunks, index, embedding_model
        chunks = chunk_text(all_extracted_text)
        index, embedding_model = create_faiss_index(chunks, embedding_model_name=EMBEDDING_MODEL_NAME)
        # کش و وضعیت PDF فقط برای این کاربر ذخیره شود
        # (در این نسخه ساده، کش کلی استفاده می‌شود. برای هر کاربر می‌توان مشابه همین ساختار را پیاده کرد)
        return "پرونده دندانپزشکی با موفقیت آپلود و پردازش شد."
    return "فایل نامعتبر است! لطفاً فایل PDF معتبر آپلود کنید.", 400

@app.route("/chat", methods=["POST"])
def chat():
  global chunks, index, embedding_model
  user_query = request.json.get("message")
  session_id = request.cookies.get('session_id')
  if not session_id:
    session_id = str(uuid.uuid4())
  if not user_query:
    return jsonify({"answer": "لطفاً یک پیام وارد کنید."})
  history = load_user_history(session_id)
  history.append({"role": "user", "content": user_query})
  recent_history = history[-HISTORY_TURNS*2:]
  history_text = ""
  for turn in recent_history:
    if turn["role"] == "user":
      history_text += f"مشتری: {turn['content']}\n"
    else:
      history_text += f"دستیار فروشگاه: {turn['content']}\n"
  user_pdf_folder = get_user_pdf_folder(session_id)
  pdf_files = [f for f in os.listdir(user_pdf_folder) if f.lower().endswith('.pdf')]
  has_pdf = len(pdf_files) > 0
  if has_pdf:
    try:
      context = retrieve_relevant_chunks(user_query, index, embedding_model, chunks, k=10)
    except Exception as e:
      print(f"Error retrieving chunks: {e}")
      context = ""
    prompt = f"""شما دستیار هوشمند فروشگاه آنلاین شاپ هستید که وظیفه‌تان کمک به مشتری درباره سفارش یا محصولات آپلودشده است.\n- اگر مشتری اطلاعاتی درباره خودش (مثل نام یا سوابق خرید) در چت قبلی داده، آن را به خاطر بسپار و در پاسخ‌ها استفاده کن.\n- اگر مشتری احوال‌پرسی یا صحبت غیرتخصصی کرد، با روابط عمومی بالا و لحن دوستانه پاسخ بده.\n- اگر سوال تخصصی درباره سفارش یا محصولات بود، فقط بر اساس متن مرجع پاسخ بده.\n- اگر اطلاعات لازم در تاریخچه چت بود، از همان استفاده کن.\n- دیگر نیازی به معرفی مجدد فروشگاه نیست، فقط روی گفتگو تمرکز کن.\n- اگر قبلاً سلام یا احوال‌پرسی در تاریخچه چت وجود دارد، دیگر سلام یا احوال‌پرسی تکرار نکن.\n- هدف این است که مشتری حس کند با یک دستیار فروشگاه واقعی و باحافظه و خوش‌برخورد گفتگو می‌کند.\n\nتاریخچه چت:\n{history_text}\nمتن مرجع:\n{context}\n\nسوال جدید مشتری:\n{user_query}\n\nپاسخ:"""
  else:
    context = ""
    if len(history) == 1:
      prompt = f"""شما دستیار هوشمند فروشگاه آنلاین شاپ هستید. فقط در این پیام خود را به عنوان نماینده فروشگاه معرفی کن و توضیحاتی از خدمات و ویژگی‌های فروشگاه ارائه بده. از پیام بعدی دیگر نیازی به معرفی مجدد خود یا فروشگاه نیست و فقط روی گفتگو و جمع‌آوری اطلاعات مشتری تمرکز کن.\nاگر قبلاً سلام یا احوال‌پرسی در تاریخچه چت وجود دارد، دیگر سلام یا احوال‌پرسی تکرار نکن.\n\nفروشگاه آنلاین شاپ ویژگی‌ها و خدمات زیر را دارد:\n- ارسال سریع و مطمئن به سراسر کشور\n- تضمین اصالت و کیفیت کالاها\n- پشتیبانی ۲۴ ساعته و مشاوره خرید\n- تنوع بی‌نظیر محصولات با قیمت مناسب\n- امکان پیگیری سفارش و مرجوعی آسان\n\nسپس وظیفه‌تان جمع‌آوری اطلاعات اولیه از مشتری قبل از ثبت سفارش است:\n- با پرسیدن سوالات مرحله‌ای (نام، شماره تماس، آدرس، نوع محصول مورد نظر، تعداد، روش ارسال و ...) اطلاعات لازم را جمع‌آوری کن\n- اگر اطلاعات کافی جمع‌آوری شد، یک خلاصه سفارش اولیه و پیشنهادات یا راهنمایی‌های خود را به مشتری ارائه بده\n- اگر مشتری احوال‌پرسی یا صحبت غیرتخصصی کرد، با روابط عمومی بالا و لحن دوستانه پاسخ بده\n- هدف این است که مشتری حس کند با یک دستیار فروشگاه واقعی و دلسوز گفتگو می‌کند\n\nسوال یا پیام جدید مشتری:\n{user_query}\n\nپاسخ:"""
    else:
      prompt = f"""شما دستیار هوشمند فروشگاه آنلاین شاپ هستید که وظیفه‌تان جمع‌آوری اطلاعات اولیه از مشتری قبل از ثبت سفارش است. دیگر نیازی به معرفی مجدد خود یا فروشگاه نیست و فقط روی گفتگو و جمع‌آوری اطلاعات مشتری تمرکز کن.\nاگر قبلاً سلام یا احوال‌پرسی در تاریخچه چت وجود دارد، دیگر سلام یا احوال‌پرسی تکرار نکن.\n- با پرسیدن سوالات مرحله‌ای (نام، شماره تماس، آدرس، نوع محصول مورد نظر، تعداد، روش ارسال و ...) اطلاعات لازم را جمع‌آوری کن.\n- اگر اطلاعات کافی جمع‌آوری شد، یک خلاصه سفارش اولیه و پیشنهادات یا راهنمایی‌های خود را به مشتری ارائه بده.\n- اگر مشتری احوال‌پرسی یا صحبت غیرتخصصی کرد، با روابط عمومی بالا و لحن دوستانه پاسخ بده.\n- هدف این است که مشتری حس کند با یک دستیار فروشگاه واقعی و دلسوز گفتگو می‌کند.\n\nتاریخچه چت:\n{history_text}\n\nسوال یا پیام جدید مشتری:\n{user_query}\n\nپاسخ:"""
  answer = get_local_completion(prompt, model_pipeline, max_tokens=1000)
  history.append({"role": "assistant", "content": answer})
  save_user_history(session_id, history)
  return jsonify({"answer": answer})

# --- Endpoint برای دانلود خلاصه پرونده پزشکی ---
@app.route("/download_summary", methods=["GET"])
def download_summary():
    session_id = request.cookies.get('session_id')
    if not session_id:
        return "Session not found", 400
    history = load_user_history(session_id)
    # استخراج اطلاعات مهم از تاریخچه
    summary_lines = []
    for turn in history:
        if turn["role"] == "user":
            summary_lines.append(f"پاسخ بیمار: {turn['content']}")
        elif turn["role"] == "assistant":
            summary_lines.append(f"پاسخ دستیار: {turn['content']}")
    summary_text = "\n".join(summary_lines)
    # فایل متنی را به صورت دانلودی ارائه بده
    from flask import Response
    return Response(summary_text, mimetype='text/plain', headers={"Content-Disposition": "attachment;filename=dental_summary.txt"})

@app.route("/download_summary_pdf", methods=["GET"])
def download_summary_pdf():
  session_id = request.cookies.get('session_id')
  if not session_id:
    return "Session not found", 400
  history = load_user_history(session_id)
  history_text = ""
  for turn in history:
    if turn["role"] == "user":
      history_text += f"کاربر: {turn['content']}\n"
    else:
      history_text += f"دستیار: {turn['content']}\n"
  summary_prompt = f"""
شما دستیار هوشمند فروشگاه آنلاین شاپ هستید. بر اساس مکالمات زیر با مشتری، یک گزارش خلاصه‌شده و ساختاریافته برای مدیریت فروشگاه تهیه کن که شامل موارد زیر باشد:
- اطلاعات پایه مشتری (در صورت وجود: نام، شماره تماس، آدرس و ...)
- شرح سفارش و محصولات مورد نیاز
- سوابق خرید و درخواست‌های مشتری
- جمع‌بندی و نتیجه‌گیری اولیه (پیشنهادات، راهنمایی‌ها، وضعیت سفارش)
- توصیه به مشتری (در صورت نیاز)

مکالمات:
{history_text}

گزارش ساختاریافته برای مدیریت فروشگاه:
"""
  summary_text = get_local_completion(summary_prompt, model_pipeline, max_tokens=800)
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font('Arial', '', 13)
  pdf.multi_cell(0, 10, summary_text)
  pdf_output = pdf.output(dest='S').encode('latin1')
  from flask import Response
  return Response(pdf_output, mimetype='application/pdf', headers={"Content-Disposition": "attachment;filename=dental_summary.pdf"})

@app.route("/download_summary_txt", methods=["GET"])
def download_summary_txt():
  session_id = request.cookies.get('session_id')
  if not session_id:
    return "Session not found", 400
  history = load_user_history(session_id)
  history_text = ""
  for turn in history:
    if turn["role"] == "user":
      history_text += f"کاربر: {turn['content']}\n"
    else:
      history_text += f"دستیار: {turn['content']}\n"
  summary_prompt = f"""
شما دستیار هوشمند فروشگاه آنلاین شاپ هستید. بر اساس مکالمات زیر با مشتری، یک گزارش خلاصه‌شده و ساختاریافته و مرتب برای مدیریت فروشگاه تهیه کن که شامل بخش‌های زیر باشد:
1. اطلاعات پایه مشتری (در صورت وجود: نام، شماره تماس، آدرس و ...)
2. شرح سفارش و محصولات مورد نیاز
3. سوابق خرید و درخواست‌های مشتری
4. جمع‌بندی و نتیجه‌گیری اولیه (پیشنهادات، راهنمایی‌ها، وضعیت سفارش)
5. توصیه به مشتری (در صورت نیاز)
گزارش را با تیتر هر بخش و با فاصله و نظم مناسب بنویس.

مکالمات:
{history_text}

گزارش ساختاریافته و مرتب برای مدیریت فروشگاه:
"""
  summary_text = get_local_completion(summary_prompt, model_pipeline, max_tokens=800)
  from flask import Response
  return Response(summary_text, mimetype='text/plain', headers={"Content-Disposition": "attachment;filename=dental_summary.txt"})

@app.route("/download_summary_html", methods=["GET"])
def download_summary_html():
  session_id = request.cookies.get('session_id')
  if not session_id:
    return "Session not found", 400
  history = load_user_history(session_id)
  history_text = ""
  for turn in history:
    if turn["role"] == "user":
      history_text += f"مشتری: {turn['content']}\n"
    else:
      history_text += f"دستیار فروشگاه: {turn['content']}\n"
  summary_prompt = f"""
شما دستیار هوشمند فروشگاه آنلاین شاپ هستید. بر اساس مکالمات زیر با مشتری، یک گزارش خلاصه‌شده و ساختاریافته و مرتب برای مدیریت فروشگاه تهیه کن که شامل بخش‌های زیر باشد:
1. اطلاعات پایه مشتری (در صورت وجود: نام، شماره تماس، آدرس و ...)
2. شرح سفارش و محصولات مورد نیاز
3. سوابق خرید و درخواست‌های مشتری
4. جمع‌بندی و نتیجه‌گیری اولیه (پیشنهادات، راهنمایی‌ها، وضعیت سفارش)
5. توصیه به مشتری (در صورت نیاز)
گزارش را با تیتر هر بخش و با فاصله و نظم مناسب بنویس و از تگ‌های HTML (مانند <h2>، <h3>، <ul>، <li>، <p>) برای ساختاربندی استفاده کن. نیازی به تگ <html> و <body> نیست.

مکالمات:
{history_text}

گزارش ساختاریافته و مرتب برای مدیریت فروشگاه (در قالب HTML):
"""
  summary_html = get_local_completion(summary_prompt, model_pipeline, max_tokens=900)
  style = '''<style>\nbody, html { background: #f7fafd; direction: rtl; font-family: Tahoma, Vazirmatn, Arial, sans-serif; color: #222; margin: 0; padding: 0; }\n.report-container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 18px; box-shadow: 0 4px 24px rgba(0,128,128,0.10); padding: 32px 28px 24px 28px; border: 2px solid #b2dfdb; }\nh1, h2, h3 { color: #ff9800; margin-top: 18px; margin-bottom: 8px; font-family: inherit; }\nh1 { font-size: 28px; text-align: center; border-bottom: 2px solid #b2dfdb; padding-bottom: 10px; margin-bottom: 24px; }\nh2 { font-size: 22px; border-right: 4px solid #ff9800; padding-right: 8px; }\nh3 { font-size: 18px; }\nul { padding-right: 24px; margin-bottom: 12px; }\nli { margin-bottom: 6px; }\np { font-size: 16px; line-height: 2; margin-bottom: 10px; }\n.section { margin-bottom: 28px; }\n@media (max-width: 800px) { .report-container { max-width: 98vw; padding: 10px 2vw; } h1 { font-size: 22px; } h2 { font-size: 18px; } }\n</style>'''
  html_report = f"""
<!DOCTYPE html>
<html lang='fa'>
<head>
<meta charset='utf-8'>
<title>گزارش خلاصه سفارش مشتری - فروشگاه آنلاین شاپ</title>
{style}
</head>
<body>
<div class='report-container'>
<h1>گزارش خلاصه سفارش مشتری - فروشگاه آنلاین شاپ</h1>
{summary_html}
</div>
</body>
</html>
"""
  from flask import Response
  return Response(html_report, mimetype='text/html', headers={"Content-Disposition": "attachment;filename=shop_summary.html"})
# تغییر دکمه دانلود در HTML به PDF
HTML = HTML.replace(
    '/download_summary',
    '/download_summary_pdf'
)

# تغییر دکمه دانلود در HTML به txt
HTML = HTML.replace(
    '/download_summary_pdf',
    '/download_summary_txt'
)

# تغییر دکمه دانلود در HTML به html
HTML = HTML.replace(
    '/download_summary_txt',
    '/download_summary_html'
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)