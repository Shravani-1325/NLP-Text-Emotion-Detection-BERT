import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import os
from dotenv import load_dotenv

load_dotenv()
BOOKS_API_KEY = os.getenv("BOOKS_API_KEY")

st.set_page_config(page_title="EmotiSense", layout="centered", page_icon="🌸")

# ============================================================
#  CSS 
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Nunito:wght@300;400;600;700&display=swap');

/* ---------- ROOT ---------- */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* ---------- ANIMATED PINK SILK BACKGROUND ---------- */
.stApp {
    background: linear-gradient(130deg, #f9c6cb 0%, #fce4ec 25%, #f48fb1 50%, #fce4ec 75%, #f9c6cb 100%);
    background-size: 300% 300%;
    animation: silkShift 10s ease infinite;
    min-height: 100vh;
}

@keyframes silkShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ---------- STREAMLIT CHROME ---------- */
footer { visibility: hidden; }

/* Remove black header bar, keep Deploy + menu transparent */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"] {
    background: transparent !important;
    box-shadow: none !important;
}

.block-container {
    padding: 2.5rem 1.5rem 3rem !important;
    max-width: 780px !important;
}

/* ---------- GLASS CARD (reusable) ---------- */
.glass {
    background: rgba(255, 255, 255, 0.30);
    border: 1.5px solid rgba(255, 255, 255, 0.60);
    border-radius: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(190, 80, 100, 0.15),
                inset 0 1px 0 rgba(255,255,255,0.55);
    padding: 30px 36px;
    margin-bottom: 22px;
}

/* ---------- HEADER ---------- */
.header-glass {
    background: rgba(255, 255, 255, 0.32);
    border: 1.5px solid rgba(255, 255, 255, 0.65);
    border-radius: 24px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 32px rgba(190, 80, 100, 0.15),
                inset 0 1px 0 rgba(255,255,255,0.55);
    padding: 36px 40px 28px;
    margin-bottom: 22px;
    text-align: center;
}

.header-glass h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #5c2237;
    letter-spacing: 1px;
    line-height: 1.15;
    margin: 0 0 6px;
}

.header-glass .tagline {
    font-size: 1.3rem;
    font-weight: 700;
    color: #7b3350;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

.header-glass .subdesc {
    font-size: 0.9rem;
    font-weight: 400;
    color: #b06878;
    letter-spacing: 0.5px;
}

/* --------------WHATS IN YOUR MIND--------------*/
.st-emotion-cache-rsr9ey p{
    font-size: large;
}

/* ---------- TEXTAREA LABEL ---------- */
.stTextArea > label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #5c2237 !important;
    margin-bottom: 8px !important;
}

/* ---------- TEXTAREA ITSELF ---------- */
.stTextArea textarea {
    background: rgba(255, 240, 244, 0.85) !important;
    border: 1.5px solid rgba(255, 200, 210, 0.80) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(12px) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    color: #4a1f30 !important;
    padding: 16px 20px !important;
    box-shadow: inset 0 2px 8px rgba(190,80,100,0.06) !important;
    transition: border-color 0.25s, box-shadow 0.25s;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: rgba(210, 90, 115, 0.80) !important;
    box-shadow: 0 0 0 3px rgba(210,90,115,0.18),
                inset 0 2px 8px rgba(190,80,100,0.08) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder {
    color: #c898a8 !important;
    font-style: italic;
}

/* ---------- BUTTONS ROW ---------- */
div.stButton {
    margin: 6px 0 8px;
}

/* Both buttons: same height, full width of their column, pill shape */
div.stButton > button {
    width: 100% !important;
    height: 50px !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    transition: transform 0.22s, box-shadow 0.22s !important;
    cursor: pointer !important;
    padding: 0 20px !important;
}

/* Analyze — pink gradient */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #e8687f 0%, #f4a8b8 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 6px 24px rgba(220,90,115,0.40) !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 12px 34px rgba(220,90,115,0.50) !important;
}
div.stButton > button:first-child:active {
    transform: translateY(0) scale(0.99) !important;
}

/* Clear — ghost style */
div.stButton > button:first-child ~ * ,
div[data-testid="column"]:nth-child(3) div.stButton > button {
    background: rgba(255,240,244,0.75) !important;
    color: #c0566a !important;
    border: 1.8px solid rgba(200,100,120,0.45) !important;
    box-shadow: none !important;
}
div[data-testid="column"]:nth-child(3) div.stButton > button:hover {
    background: rgba(255,210,220,0.85) !important;
    border-color: rgba(200,80,100,0.65) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(200,80,100,0.20) !important;
}

/* ---------- RESULT ROW ---------- */
.result-row {
    display: flex;
    gap: 20px;
    margin-bottom: 22px;
}

.result-card {
    flex: 1;
    background: rgba(255, 255, 255, 0.30);
    border: 1.5px solid rgba(255, 255, 255, 0.60);
    border-radius: 22px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 28px rgba(190, 80, 100, 0.14),
                inset 0 1px 0 rgba(255,255,255,0.55);
    padding: 26px 28px;
}

.result-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    color: #5c2237;
    margin: 0 0 4px;
}

.result-card .conf-text {
    font-size: 0.92rem;
    color: #9a5060;
    margin-bottom: 14px;
}

.result-card .tip-line {
    font-size: 0.95rem;
    color: #6a3045;
    line-height: 1.65;
    margin-bottom: 5px;
    font-style : italic;
}

.result-card .quote-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #5c2237;
    margin-bottom: 14px;
}

.result-card .quote-body {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-style: italic;
    color: #7b3350;
    line-height: 1.7;
}

/* ---------- BOOKS SECTION ---------- */
.books-glass {
    background: rgba(255, 255, 255, 0.30);
    border: 1.5px solid rgba(255, 255, 255, 0.60);
    border-radius: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(190, 80, 100, 0.14),
                inset 0 1px 0 rgba(255,255,255,0.55);
    padding: 28px 32px;
    margin-bottom: 22px;
}

.books-glass h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    color: #5c2237;
    text-align: center;
    margin-bottom: 22px;
    letter-spacing: 0.5px;
}

.book-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 20px;
}

.book-item {
    text-align: center;
    width: 130px;
}

.book-item img {
    width: 120px;
    height: 175px;
    object-fit: cover;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(150, 60, 80, 0.25);
    transition: transform 0.25s, box-shadow 0.25s;
    display: block;
    margin: 0 auto;
}

.book-item img:hover {
    transform: translateY(-5px) scale(1.06);
    box-shadow: 0 14px 30px rgba(150, 60, 80, 0.35);
}

.book-item .book-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #5c2237;
    margin-top: 10px;
    line-height: 1.3;
    word-break: break-word;
}

/* ---------- SPINNER ---------- */
.stSpinner > div { border-top-color: #e8687f !important; }

/* ---------- WARNING ---------- */
.stAlert {
    background: rgba(255,255,255,0.30) !important;
    border-radius: 16px !important;
    border: 1.5px solid rgba(255,255,255,0.60) !important;
    backdrop-filter: blur(14px) !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
#  MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained(
        "shravani1305/bert-emotion-model"
    )

    tokenizer = BertTokenizer.from_pretrained(
        "shravani1305/bert-emotion-model"
    )

    return model, tokenizer

model, tokenizer = load_model()

LABELS = [
    'anger', 'empty', 'enthusiasm', 'fun', 'happiness',
    'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise'
]

# ============================================================
#  CONTENT
# ============================================================
QUOTES = {
    "happiness": [
        "Let your joy ripple outward — even the smallest light can brighten someone’s darkest moment 🌟",
        "Happiness is not something to hold, but something to share — let it flow through you 💛",
        "Stay in this moment… this is what peace feels like, this is what you’ve been waiting for 🌻"
    ],

    "sadness": [
        "It’s okay to sit with your sadness — not every feeling needs fixing, some just need understanding 💙",
        "Storms don’t ask permission to arrive, but they always leave — hold on, this too will pass 🌧️",
        "Be gentle with yourself today… healing doesn’t happen all at once, it unfolds slowly 🤍"
    ],

    "anger": [
        "Anger speaks loudly, but clarity speaks softly — pause long enough to hear it 🌬️",
        "You don’t need to react to everything you feel — sometimes strength is choosing stillness 🧘",
        "Your calm is not weakness, it is control — and control is real power 💪"
    ],

    "love": [
        "Love is not just something you feel, it’s something you choose — again and again ❤️",
        "In a world that rushes, take time to truly feel and give love — that’s where life lives 🌹",
        "Love deeply, but don’t lose yourself — the right love never asks you to shrink 🦋"
    ],

    "neutral": [
        "There is beauty in stillness — not every moment needs to be loud to be meaningful 🌿",
        "Balance is not something you find, it’s something you create within yourself ⚖️",
        "Peace is quiet, subtle, and steady — learn to recognize it in simple moments 🕊️"
    ],

    "fun": [
        "Life isn’t meant to be lived seriously all the time — leave space for laughter 🎉",
        "Joy hides in playful moments — don’t rush past them looking for something bigger 😄",
        "Sometimes the best thing you can do is let go… and just enjoy being alive 🎈"
    ],

    "enthusiasm": [
        "This energy you feel — don’t waste it doubting yourself, use it to build something bold 🚀",
        "Your excitement is a signal — it’s pointing you toward something that matters ⚡",
        "Go forward with fire in your heart, but keep your mind steady — that’s how you win 🔥"
    ],

    "relief": [
        "You made it through something heavy — take a moment to truly breathe and feel that release 😌",
        "Not everything stays forever — and this lightness you feel is proof of that 🍃",
        "Rest here for a while… you don’t always have to be in survival mode 🌈"
    ],

    "surprise": [
        "Life doesn’t always follow logic — and sometimes that’s where the magic hides ✨",
        "Stay open to the unexpected — not everything unknown is something to fear 🌟",
        "Moments of surprise remind us — life is still unfolding, still alive 🎁"
    ],

    "hate": [
        "Holding onto hate is like carrying weight that was never yours to begin with 🕊️",
        "Let go, not because they deserve it — but because you deserve peace 🌊",
        "The more you release what drains you, the more space you create for something better 🌸"
    ],

    "empty": [
        "Feeling empty doesn’t mean you are empty — it means you’ve been carrying too much for too long 🤍",
        "This quiet inside you is not the end — it’s the space where something new can begin 🌅",
        "Even when you feel nothing, you still matter — more than you realize 💜"
    ],
}

EMOJI = {
    "happiness": "😄", "sadness": "😢",  "anger":      "😡",
    "love":      "😍", "neutral": "😐",  "fun":        "🎉",
    "enthusiasm":"🚀", "relief":  "😌",  "surprise":   "😲",
    "hate":      "😤", "empty":   "🫥",
}

TIPS = {
    "happiness":  ["💬 Keep spreading your light", "💛 Share your joy with someone"],
    "sadness":    ["💙 It's okay to feel low", "🤍 Reach out to someone you trust"],
    "anger":      ["🌬️ Try box breathing", "🧘 Step away and reset"],
    "love":       ["❤️ Express how you feel", "🌹 Cherish this feeling"],
    "neutral":    ["🌿 Ground yourself with nature", "📖 Reflect in a journal"],
    "fun":        ["🎉 Embrace this playful energy", "😄 Share the fun!"],
    "enthusiasm": ["🚀 Channel this into action", "⚡ Set a bold goal today"],
    "relief":     ["😌 Rest and recharge", "🍃 You've earned this calm"],
    "surprise":   ["✨ Stay curious", "🎁 Explore what surprised you"],
    "hate":       ["🕊️ Release what you can't control", "🌊 Try a grounding exercise"],
    "empty":      ["🤍 Talk to someone you trust", "🌅 One small step forward"],
}


# ============================================================
#  PREDICT
# ============================================================
def predict(text: str):
    inputs = tokenizer(
        [text], padding=True, truncation=True,
        max_length=128, return_tensors="tf"
    )
    logits = model(inputs).logits
    pred   = int(np.argmax(logits, axis=1)[0])
    probs  = tf.nn.softmax(logits)[0]
    return LABELS[pred], float(probs[pred])


# ============================================================
#  BOOKS
# ============================================================
def get_books(emotion: str):
    books = []
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={emotion}&key={BOOKS_API_KEY}&maxResults=6"
        res = requests.get(url, timeout=8)
        data = res.json()

        for b in data.get("items", []):
            info = b.get("volumeInfo", {})
            img  = (info.get("imageLinks") or {}).get("thumbnail") or \
                   (info.get("imageLinks") or {}).get("smallThumbnail")
            if img:
                # force https so browser doesn't block mixed content
                img = img.replace("http://", "https://")
                books.append({
                    "title": info.get("title", "No Title"),
                    "img":   img,
                })
            if len(books) == 4:
                break
    except Exception as e:
        st.error(f"Books API error: {e}")
    return books


# ============================================================
#  UI — HEADER
# ============================================================
st.markdown("""
<div class="header-glass">
    <h1>🌸 EmotiSense</h1>
    <div class="tagline">Feel. Understand. Heal.</div>
    <div class="subdesc">EmotiSense is an AI-powered Text Emotion Analysis Detection Web App that analyzes user text using a fine-tuned BERT model and recommends mood-based books and motivational quotes through a modern Streamlit interface.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  UI — INPUT
# ============================================================
# Session state for clear functionality
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def clear_input():
    st.session_state.input_text = ""
    st.session_state.show_result = False

if "show_result" not in st.session_state:
    st.session_state.show_result = False

text = st.text_area("💬 What's on your mind?", height=130,
                    placeholder="Type how you're feeling…",
                    value=st.session_state.input_text,
                    key="input_text")

col_left, col_btn, col_clear, col_right = st.columns([1, 2, 2, 1], gap="small")
with col_btn:
    btn = st.button("✨ Analyze Emotion", use_container_width=True)
with col_clear:
    clear = st.button("🗑️ Clear", on_click=clear_input, use_container_width=True)

# ============================================================
#  UI — RESULT
# ============================================================
if btn and text.strip():
    st.session_state.show_result = True

if st.session_state.get("show_result") and text.strip():

    with st.spinner("Analyzing your emotion…"):
        emotion, conf = predict(text)

    tips  = TIPS.get(emotion, [])
    quote = np.random.choice(QUOTES.get(emotion, ["Stay strong 💪"]))

    tips_html = "".join(f'<div class="tip-line">• {t}</div>' for t in tips)

    st.markdown(f"""
    <div class="result-row">
        <div class="result-card">
            <h3>{EMOJI[emotion]} {emotion.capitalize()}</h3>
            <div class="conf-text">Confidence: {conf:.2f}</div>
            {tips_html}
        </div>
        <div class="result-card">
            <div class="quote-label">🌈 Quote for You</div>
            <div class="quote-body">"{quote}"</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Books — use components.html to prevent Streamlit escaping & in Google URLs
    import streamlit.components.v1 as components

    with st.spinner("Fetching book recommendations…"):
        books = get_books(emotion)

    if books:
        book_items_html = "".join(
            f'''<div class="book-item">
                <img src="{b["img"]}" alt="{b["title"][:25]}">
                <div class="book-title">{b["title"][:30]}</div>
            </div>'''
            for b in books
        )
    else:
        book_items_html = '''<p style="text-align:center;color:#9a5060;font-size:0.95rem;width:100%;">
            Could not load books. Check your BOOKS_API_KEY in .env</p>'''

    components.html(
        f"""<!DOCTYPE html><html><head>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Nunito:wght@600&display=swap" rel="stylesheet">
        <style>
        body{{margin:0;padding:0;background:transparent;}}
        .books-glass{{background:rgba(252,228,236,0.55);border:1.5px solid rgba(255,255,255,0.60);
            border-radius:24px;backdrop-filter:blur(20px);padding:28px 32px;
            box-shadow:0 8px 32px rgba(190,80,100,0.14);}}
        h3{{font-family:'Playfair Display',serif;font-size:1.25rem;color:#5c2237;
            text-align:center;margin:0 0 20px;letter-spacing:0.5px;}}
        .book-row{{display:flex;justify-content:center;flex-wrap:wrap;gap:20px;}}
        .book-item{{text-align:center;width:128px;}}
        .book-item img{{width:118px;height:172px;object-fit:cover;border-radius:12px;
            box-shadow:0 6px 20px rgba(150,60,80,0.30);transition:transform 0.25s,box-shadow 0.25s;
            display:block;margin:0 auto;cursor:pointer;}}
        .book-item img:hover{{transform:translateY(-5px) scale(1.06);box-shadow:0 14px 30px rgba(150,60,80,0.40);}}
        .book-title{{font-family:'Nunito',sans-serif;font-size:0.80rem;font-weight:600;color:#5c2237;
            margin-top:10px;line-height:1.35;word-break:break-word;}}
        </style></head><body>
        <div class="books-glass">
            <h3>📚 Books To Read</h3>
            <div class="book-row">{book_items_html}</div>
        </div>
        </body></html>""",
        height=330,
        scrolling=False,
    )

elif btn and not text.strip():
    st.warning("Please type something first 💬")
    
 
# ============================================================
#  FOOTER — Made By
# ============================================================
st.markdown("""
<div style="
    text-align: center;
    margin-top: 40px;
    padding: 18px 0 8px;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #9a4060;
    letter-spacing: 2px;
">
    🌸 &nbsp; Made with <span style="color:#e8687f;">♥</span> by &nbsp;<strong style="color:#5c2237; font-size:1.05rem;">Shravani More</strong> &nbsp; 🌸
</div>
""", unsafe_allow_html=True)