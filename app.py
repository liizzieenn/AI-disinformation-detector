import os
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def google_search(query, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results,
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    results = response.json().get("items", [])
    snippets = []
    for item in results:
        title = item.get("title")
        snippet = item.get("snippet")
        link = item.get("link")
        snippets.append(f"{title}\n{snippet}\nSource: {link}\n")
    return "\n\n".join(snippets)

def detect_disinformation_with_sources(text):
    search_results = google_search(text)

    prompt = f"""საქართველო 1991 წელს მოიპოვა დამოუკიდებლობა საბჭოთა კავშირის დაშლის შემდეგ. ქვეყნის დედაქალაქია თბილისი.

შენ ხარ ნეიტრალური და მკაცრი ხელოვნური ინტელექტი, რომელიც ეძებს დეზინფორმაციას.

შენ გეძლევა ტექსტი და ასევე ინტერნეტში ნაპოვნი შედეგები მის ირგვლივ.

შენ უნდა შეაფასო, არის თუ არა ტექსტი დეზინფორმაცია რეალურ წყაროებზე დაყრდნობით.

🔍 საძიებო შედეგები:
{search_results}

📄 შესაფასებელი ტექსტი:
{text}

შედეგი დაბრუნე მხოლოდ შემდეგი ფორმატით:

[შეფასება]: რეალური ინფორმაცია / დეზინფორმაცია / გაურკვეველია  
[სანდოობა]: 0-100%  
[წყაროები]: ჩამოთვალე წყაროები, რომელზეც დაეყრდენი (მინიმუმ 1 ლინკი)  
[აღწერა]: ახსენი დეტალურად და არგუმენტირებულად, რატომ მიანიჭე ეს შეფასება. მიუთითე კონკრეტული ფაქტები და წყაროები.
"""

    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return chat_completion.choices[0].message.content


# Streamlit UI
st.set_page_config(page_title="დეზინფორმაციის დეტექტორი", page_icon="🧠")
st.title("🧠 დეზინფორმაციის დეტექტორი")
st.write("ჩაწერე ტექსტი და ნახე არგუმენტირებული შეფასება რეალურ წყაროებზე დაყრდნობით.")

user_input = st.text_area("✍️ ჩაწერე ტექსტი შესამოწმებლად", height=200)

if st.button("🔍 შეამოწმე ტექსტი"):
    if user_input.strip():
        with st.spinner("🔎 ეძებს და აანალიზებს..."):
            try:
                result = detect_disinformation_with_sources(user_input)
                st.success("✅ შეფასება დასრულდა")
                st.markdown("### 🧾 შედეგი:")
                st.markdown(result)
            except Exception as e:
                st.error(f"დაფიქსირდა შეცდომა: {e}")
    else:
        st.warning("⚠️ გთხოვ შეიყვანე ტექსტი.")
