import os
import time
import logging
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
from groq import Groq

# Logging Setup 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API Keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

REQUIRED_KEYS = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "GOOGLE_CSE_ID": GOOGLE_CSE_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
    "NEWS_API_KEY": NEWS_API_KEY,
    "GNEWS_API_KEY": GNEWS_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN
}
missing = [k for k, v in REQUIRED_KEYS.items() if not v]
if missing:
    raise EnvironmentError(f"Missing API keys: {', '.join(missing)}")

# Initialize Groq
client = Groq(api_key=GROQ_API_KEY)


# Google Search 
def google_search(query, num_results=3, retries=3, backoff=1):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results
    }
    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
            items = res.json().get("items", [])
            return [{
                "title": item.get("title", "სათაური არ არის"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            } for item in items]
        except requests.exceptions.RequestException as e:
            logging.warning(f"Google Search შეცდომა (ცდა {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    return []


# NewsAPI Search 
def newsapi_search(query, language="ka", page_size=3):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        articles = res.json().get("articles", [])
        return [{
            "title": a["title"],
            "link": a["url"],
            "description": a.get("description", "")
        } for a in articles]
    except Exception as e:
        logging.warning(f"NewsAPI შეცდომა: {e}")
        return []


# GNews Search 
def gnews_search(query, max_results=3):
    try:
        url = f"https://gnews.io/api/v4/search?q={query}&lang=ka&token={GNEWS_API_KEY}&max={max_results}"
        res = requests.get(url)
        res.raise_for_status()
        articles = res.json().get("articles", [])
        return [{
            "title": a["title"],
            "link": a["url"],
            "description": a.get("description", "")
        } for a in articles]
    except Exception as e:
        logging.warning(f"GNews API შეცდომა: {e}")
        return []


# Parse Score 
def parse_score_and_label(ai_text):
    match = re.search(r"(\d{1,3})%", ai_text)
    score = int(match.group(1)) if match else 50
    if "რეალური" in ai_text:
        label = "რეალური ინფორმაცია"
    elif "დეზინფორმაცია" in ai_text:
        label = "დეზინფორმაცია"
    else:
        label = "გაურკვეველია"
    return score, label


# Show Pie 
def show_score_pie(score):
    labels = ['რეალური ინფორმაცია', 'დეზინფორმაცია']
    sizes = [score, 100 - score]
    colors = ['#4CAF50', '#F44336']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)


# AI Analysis
def detect_disinformation_with_sources(text):
    google_results = google_search(text)
    news_results = newsapi_search(text)
    gnews_results = gnews_search(text)

    has_sources = any([google_results, news_results, gnews_results])

    if has_sources:
        combined_sources = (
            f"🔍 Google შედეგები:\n{google_results}\n\n🗞 NewsAPI სიახლეები:\n{news_results}\n\n📰 GNews სტატიები:\n{gnews_results}"
        )
        prompt = f"""
შენ ხარ ნეიტრალური, მკაცრი და აკადემიური ქართული ენის ხელოვნური ინტელექტი.

შეაჯამე ქვემოთ მოყვანილი წყაროები და შეაფასე ტექსტი — დეზინფორმაციაა თუ არა.

წყაროები:
{combined_sources}

📄 შესაფასებელი ტექსტი:  
{text}

შედეგი ფორმატით:

📊 შეფასება: რეალური ინფორმაცია / დეზინფორმაცია / გაურკვეველია  
🔢 დონე: 0-100%  
🧾 არგუმენტირებული აღწერა  
🔗 წყაროები
"""
    else:
        prompt = f"""
შენ ხარ ნეიტრალური, მკაცრი და აკადემიური ქართული ენის ხელოვნური ინტელექტი.

წყაროები ვერ მოიძებნა, შეაფასე მხოლოდ ტექსტის საფუძველზე:

📄 შესაფასებელი ტექსტი:  
{text}

შედეგი ფორმატით:

📊 შეფასება: რეალური ინფორმაცია / დეზინფორმაცია / გაურკვეველია  
🔢 დონე: 0-100%  
🧾 არგუმენტირებული აღწერა  
🔗 წყაროები: "სანდო ღია წყარო ვერ მოიძებნა"
"""
    try:
        res = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        ai_output = res.choices[0].message.content
        return ai_output, google_results, news_results, gnews_results
    except Exception as e:
        logging.error(f"Groq API შეცდომა: {e}")
        return "შეცდომა მოხდა შეფასების პროცესში.", [], [], []


# Streamlit UI 
st.set_page_config(page_title="დეზინფორმაციის დეტექტორი", page_icon="🧠")
st.title("🧠 დეზინფორმაციის დეტექტორი")
st.markdown("**ჩაწერე ტექსტი და შეამოწე, არის თუ არა დეზინფორმაცია.**")

user_input = st.text_area("✍️ ტექსტი შესამოწმებლად (მაქს. 1000 სიმბოლო)", max_chars=1000, height=150)

if st.button("🔍 შეამოწე ტექსტი"):
    if user_input.strip():
        with st.spinner("🔎 ეძებს წყაროებს და აანალიზებს..."):
            ai_result, google_results, news_results, gnews_results = detect_disinformation_with_sources(user_input)

        st.success("✅ შეფასება დასრულდა")

        st.markdown("### 📄 AI შეფასება:")
        st.markdown(ai_result)

        score, label = parse_score_and_label(ai_result)
        st.markdown(f"**შეფასება:** {label} ({score}%)")

        show_score_pie(score)
        st.markdown("---")


        def show_sources_table(results, title):
            if results:
                df = pd.DataFrame(results)
                df['title'] = df['title'].apply(lambda x: f"[{x}]")
                df = df.rename(
                    columns={'title': 'სათაური', 'link': 'ლინკი', 'snippet': 'აღწერა', 'description': 'აღწერა'})
                st.markdown(f"### 🔗 {title} წყაროები:")
                for _, row in df.iterrows():
                    st.markdown(f"**[{row['სათაური'][1:-1]}]({row['ლინკი']})**  \n📄 {row.get('აღწერა', '')}")
            else:
                st.markdown(f"### 🔗 {title} წყაროები: \n_შედეგი ვერ მოიძებნა._")


        show_sources_table(google_results, "Google-ში")
        show_sources_table(news_results, "NewsAPI-ზე")
        show_sources_table(gnews_results, "GNews-ზე")
    else:
        st.warning("⚠️ გთხოვ შეიყვანე ტექსტი შესამოწმებლად.")
