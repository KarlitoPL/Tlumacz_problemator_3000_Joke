import streamlit as st
import requests
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import dotenv_values
from openai import OpenAI


env = dotenv_values(".env")

openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

# Połączenie z serwerem Qdrant 
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"],
    api_key=env["QDRANT_API_KEY"],
)

def create_account_collection(account_name):
    """
    Sprawdza, czy kolekcja (konto) o danej nazwie już istnieje.
    Jeśli nie tworzy nową kolekcję w Qdrant.
    """
    existing = get_qdrant_client.get_collections().collections
    collection_names = [col.name for col in existing]
    if account_name in collection_names:
        st.info(f"Konto '{account_name}' już istnieje.")
    else:
        get_qdrant_client.create_collection(
            collection_name=account_name,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )
        st.info(f"Konto '{account_name}' zostało utworzone.")

def save_history(account_name, record_type, prompt_text, generated_text, cost_usd, cost_pln):
    """
    Zapisuje rekord (historię) do kolekcji odpowiadającej kontu w Qdrant.
    """
    record_id = str(uuid.uuid4())
    vector = [0.0]  # Dummy wektor – w przyszłości można zastąpić właściwymi embeddingami.
    payload = {
        "type": record_type,
        "prompt": prompt_text,
        "generated_text": generated_text,
        "cost_usd": cost_usd,
        "cost_pln": cost_pln,
        "timestamp": int(time.time())
    }
    get_qdrant_client.upsert(
        collection_name=account_name,
        points=[{
            "id": record_id,
            "vector": vector,
            "payload": payload
        }]
    )
    st.success(f"Rekord zapisany pomyślnie. ID rekordu: {record_id}")

def generate_prompt(problem, mode, style=None):
    """
    Generuje prompt na podstawie wpisanego problemu oraz wybranego trybu.
    """
    if mode == "Ulubione Universum":
        prompt = ("Stwórz mi z tego problemu szczegółowe opowiadanie dla osoby, która go nie rozumie, aby mu wytłumaczyć.")
        if style:
            prompt += f" Ale w stylu opowiadań {style}."
    elif mode == "Zabawnie":
        prompt = ("Stwórz mi z tego problemu długi, kilkupoziomowy żart, ale taki śmieszny jak w try not laugh, "
                  "dla osoby, która go nie rozumie, aby mu szczegółowo i po ludzku wytłumaczyć jak, bez żadnych innych wstawek i wytłumaczeń.")
    else:
        prompt = ""
    prompt += "\nProblem: " + problem
    return prompt



def get_gpt4_response(prompt):
    """
    Wywołuje API GPT-4, aby wygenerować odpowiedź na zadany prompt.
    """
    try:
        response = openai_client.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Błąd przy generowaniu odpowiedzi: {e}")
        return None

def approximate_token_count(text):
    """
    Prosta aproksymacja liczby tokenów: średnio 4 znaki = 1 token.
    """
    return max(1, len(text) // 4)

def calculate_cost(prompt_text, generated_text, exchange_rate=4):
    """
    Oblicza przybliżony koszt wywołania API GPT-4:
      - 2.5 USD za milion tokenów dla prompta,
      - 10.0 USD za milion tokenów dla wygenerowanego tekstu.
    Zwraca koszt w USD oraz przeliczony na złote.
    """
    prompt_tokens = approximate_token_count(prompt_text)
    output_tokens = approximate_token_count(generated_text)
    cost_prompt_usd = prompt_tokens * 2.5 / 1000000
    cost_output_usd = output_tokens * 10.0 / 1000000
    total_cost_usd = cost_prompt_usd + cost_output_usd
    total_cost_pln = total_cost_usd * exchange_rate
    return total_cost_usd, total_cost_pln

# Budowanie interfejsu aplikacji w Streamlit
st.title("Nauka poprzez zabawne opowieści.")

# Dane konta
account_name = st.text_input("Podaj unikalną nazwę konta:")
password = st.text_input("Podaj hasło (opcjonalnie):", type="password")

# Wpis problemu
problem_input = st.text_area("Wpisz problem lub cokolwiek, czego nie rozumiesz:")

# Wybór trybu: Ulubione Universum lub Zabawnie
mode = st.radio("Wybierz tryb:", ("Ulubione Universum", "Zabawnie"))

# Dodatkowy input przy wyborze "Ulubione Universum"
style_input = None
if mode == "Ulubione Universum":
    style_input = st.text_input("Wpisz ulubionego pisarza, reżysera lub tytuł ulubionego filmu:")

if st.button("Generuj"):
    if not account_name:
        st.error("Proszę podać unikalną nazwę konta!")
    elif not problem_input:
        st.error("Pole problem jest puste!")
    else:
        # Utworzenie konta (kolekcji) w Qdrant, jeśli jeszcze nie istnieje
        create_account_collection(account_name)
        
        # Generowanie prompta na podstawie danych wejściowych
        prompt = generate_prompt(problem_input, mode, style_input)
        st.subheader("Wygenerowany prompt:")
        st.code(prompt, language="python")
        
        # Uzyskanie odpowiedzi z GPT-4
        generated_text = get_gpt4_response(prompt)
        if generated_text:
            
            st.subheader("Wygenerowany tekst:")
            st.write(generated_text)
            
            # Obliczanie przybliżonego kosztu wywołania OpenAI API
            cost_usd, cost_pln = calculate_cost(prompt, generated_text)
            st.markdown("---")
            st.write(f"Szacowany koszt wywołania API GPT-4: {cost_usd*100:.2f} centów USD (~{cost_pln:.2f} zł)")
            
            # Zapis historii do Qdrant
            save_history(account_name, mode, prompt, generated_text, cost_usd, cost_pln)
