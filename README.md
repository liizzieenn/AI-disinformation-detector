# AI-disinformation-detector
This is a simple AI-powered program that checks if a piece of text contains false or misleading information.

1. ბიბლიოთეკების შემოტანა
python
Copy code
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
➡️ ეს ნაწილი "ინსტრუმენტების ყუთია" – ყველა საჭირო ბიბლიოთეკას ვამატებთ, რაზეც აპი მუშაობს:

os, time, logging – ოპერაციული სისტემის და ლოგირების ფუნქციებისთვის.

requests – ინტერნეტიდან მონაცემების წამოსაღებად.

streamlit – ჩვენი ვებ-აპის UI-სთვის.

pandas, matplotlib – ცხრილებისა და დიაგრამების სანახავად.

re – ტექსტიდან მონაცემების ამოსაღებად.

dotenv – საიდუმლო API key-ების დასამალად.

groq – ჩვენი AI ტვინის გამოსაყენებლად (LLaMA3).

🛠️ 2. ლოგირების გაშვება
python
Copy code
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
➡️ როცა რამე გაფუჭდება, ეს ხაზები ამაზე ინფორმაციას console-ში გამოიტანს.

🔑 3. .env ფაილიდან API Key-ების წამოღება
python
Copy code
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
...
➡️ აქ პროგრამა _დამალული_ .env ფაილიდან იღებს API key-ებს: Google-ის, Groq-ის, NewsAPI-ს და სხვების.

❗ 4. Key-ების შემოწმება
python
Copy code
REQUIRED_KEYS = {...}
...
if missing:
    raise EnvironmentError(...)
➡️ თუ რომელიმე საჭირო key აკლია, აპი გაჩერდება და დაგიწერს, რომ რაღაც ვერ ნახა. უსაფრთხოებისთვის კარგია.

🧠 5. Groq Client-ის ჩატვირთვა
python
Copy code
client = Groq(api_key=GROQ_API_KEY)
➡️ ვუთითებთ AI-სთან მისასვლელ კავშირს. ამით მერე შეგვიძლია ტექსტი ჩავაბაროთ LLaMA3-ს შესაფასებლად.

🔎 6. Google Search ფუნქცია
python
Copy code
def google_search(query, num_results=3, retries=3, backoff=1):
    ...
➡️ ეს ფუნქცია აკეთებს რეალურ Google ძიებას მომხმარებლის ტექსტზე. აბრუნებს პირველი 3 შედეგს. თუ ვერ გაწვდა – ცდილობს ხელახლა.

📰 7. NewsAPI ძიება
python
Copy code
def newsapi_search(query, language="ka", page_size=3):
    ...
➡️ NewsAPI-ზე ეძებს ქართულ სტატიებს. აბრუნებს სათაურს, აღწერას და ლინკს.

🗞 8. GNews ძიება
python
Copy code
def gnews_search(query, max_results=3):
    ...
➡️ იგივეა, უბრალოდ სხვა სიახლეების პლატფორმიდან მოაქვს ინფორმაცია.

🧮 9. შეფასების გაშიფვრა
python
Copy code
def parse_score_and_label(ai_text):
    ...
➡️ აიღებს AI-ის პასუხს, ამოიღებს შიდა % შეფასებას (მაგ. 85%) და დაგიბრუნებს შერჩეულ “ლეიბლს”: რეალური/დეზინფორმაცია/გაურკვეველი.

📊 10. პაიჩარტის დახატვა
python
Copy code
def show_score_pie(score):
    ...
➡️ ხატავს მარტივ ფერად პაიჩარტს იმის ჩვენებისთვის, რამდენად სანდოდ ჩათვალა AI-მა ტექსტი.

🧠 11. მთავარი ანალიზის ფუნქცია
python
Copy code
def detect_disinformation_with_sources(text):
    ...
➡️ ეს ყველაფერი აქ იწყება:

ეძებს ტექსტს Google-ზე, NewsAPI-ზე და GNews-ზე.

აგროვებს შედეგებს.

ქმნის პრემიუმ პრომპტს Groq-ისთვის, სადაც წყაროებიც უდევს.

უგზავნის Groq-ს.

აბრუნებს შეფასებას.

თუ წყაროები ვერ იპოვა, აგზავნის მხოლოდ ტექსტზე დაფუძნებულ შეფასებას.

🖼️ 12. Streamlit UI – ვებ აპის ნაწილი
python
Copy code
st.set_page_config(...)
st.title("🧠 დეზინფორმაციის დეტექტორი")
...
➡️ აქ იწყება რეალური ვებგვერდის ნაწილი – რას ხედავს მომხმარებელი:

ტიტული

ტექსტის შესაყვანი ველი

ღილაკი (“შეამოწმე ტექსტი”)

🖱️ 13. ღილაკზე დაჭერის ლოგიკა
python
Copy code
if st.button("🔍 შეამოწე ტექსტი"):
    ...
➡️ თუ ტექსტი შეყვანილია და ღილაკს დააჭერ:

ეძებს წყაროებს

უგზავნის Groq-ს

აჩვენებს შეფასებას

აგენერირებს დიაგრამას

ცხრილში აჩვენებს წყაროებს

📋 14. წყაროების ცხრილებად ჩვენება
python
Copy code
def show_sources_table(results, title):
    ...
➡️ ეს ფუნქცია თითო წყაროს შედეგს აჩვენებს ვიზუალურად, სათაურით, ლინკით და აღწერით.

✅ 15. საბოლოო გაფრთხილება
python
Copy code
else:
    st.warning("⚠️ გთხოვ შეიყვანე ტექსტი შესამოწმებლად.")
➡️ თუ ცარიელი დატოვე ტექსტი, შეგახსენებს, რომ რაღაც მაინც შეიყვანო.

