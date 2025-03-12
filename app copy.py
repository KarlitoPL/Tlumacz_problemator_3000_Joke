import streamlit as st
import requests
import uuid
import time
import bcrypt
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
import openai

# -------------------------  KONFIGURACJA  -------------------------

env = dotenv_values(".env")

# Inicjalizacja klienta OpenAI
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

@st.cache_resource
def get_qdrant_client():
    """
    Zwraca zcache'owaną instancję klienta Qdrant.
    """
    return QdrantClient(
        url=env["QDRANT_URL"],     # np. "https://xxx-xxxxx-xxx-xxxxx.aws.cloud.qdrant.io"
        api_key=env["QDRANT_API_KEY"],
    )

# -------------------------  FUNKCJE: UŻYTKOWNICY  -------------------------

def create_users_collection():
    """
    Tworzy (jeśli nie istnieje) kolekcję 'users' do przechowywania danych o użytkownikach.
    """
    client = get_qdrant_client()
    existing = client.get_collections().collections
    names = [col.name for col in existing]
    if "users" not in names:
        client.create_collection(
            collection_name="users",
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )

def hash_password(password: str) -> str:
    """
    Hashuje hasło przy użyciu bcrypt.
    Zwraca zakodowany hash w formie str.
    """
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    """
    Sprawdza, czy hasło 'password' pasuje do zapisanego hasha 'hashed'.
    """
    return bcrypt.hashpw(password.encode("utf-8"), hashed.encode("utf-8")) == hashed.encode("utf-8")

def register_user(username: str, password: str) -> bool:
    """
    Rejestruje nowego użytkownika w kolekcji 'users'.
    Zwraca True, jeśli rejestracja się udała, False jeśli user istnieje.
    """
    client = get_qdrant_client()
    if find_user(username):
        return False  # użytkownik już istnieje

    user_id = str(uuid.uuid4())
    payload = {
        "username": username,
        "hashed_password": hash_password(password),
        "created_at": int(time.time())
    }
    client.upsert(
        collection_name="users",
        points=[{
            "id": user_id,
            "vector": [0.0],  # Dummy wektor
            "payload": payload
        }]
    )
    return True

def find_user(username: str):
    client = get_qdrant_client()

    all_points = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name="users",
            limit=50,
            offset=next_offset
        )
        all_points.extend(points)

        # Jeśli next_offset jest None, nie ma kolejnych "stron"
        if next_offset is None:
            break

    for point in all_points:
        payload = point.payload
        if payload.get("username") == username:
            return payload
    return None



def login_user(username: str, password: str) -> bool:
    """
    Loguje użytkownika. Zwraca True, jeśli dane poprawne, False w przeciwnym wypadku.
    """
    user = find_user(username)
    if not user:
        return False
    hashed = user["hashed_password"]
    return check_password(password, hashed)

def create_user_collection_if_not_exists(username: str):
    """
    Dla zalogowanego użytkownika tworzy kolekcję w Qdrant,
    w której będą przechowywane jego historie.
    """
    client = get_qdrant_client()
    existing = client.get_collections().collections
    collection_names = [col.name for col in existing]
    if username not in collection_names:
        client.create_collection(
            collection_name=username,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )

# -------------------------  FUNKCJE: GENERATOR HISTORII  -------------------------

def save_history(account_name, record_type, prompt_text, generated_text, cost_usd, cost_pln):
    """
    Zapisuje rekord (historię) do kolekcji odpowiadającej kontu w Qdrant.
    """
    client = get_qdrant_client()
    record_id = str(uuid.uuid4())
    vector = [0.0]
    payload = {
        "type": record_type,
        "prompt": prompt_text,
        "generated_text": generated_text,
        "cost_usd": cost_usd,
        "cost_pln": cost_pln,
        "timestamp": int(time.time())
    }
    client.upsert(
        collection_name=account_name,
        points=[{
            "id": record_id,
            "vector": vector,
            "payload": payload
        }]
    )
    st.success(f"Rekord zapisany pomyślnie. ID: {record_id}")

def get_history(account_name):
    client = get_qdrant_client()
    all_points = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=account_name,
            limit=50,
            offset=next_offset
        )
        all_points.extend(points)

        if next_offset is None:
            break

    return all_points



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
    Wywołuje API GPT-4 (lub inny model), aby wygenerować odpowiedź na zadany prompt.
    """


    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Zmień na "gpt-4" lub inny model, do którego masz dostęp
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
    cost_prompt_usd = prompt_tokens * 2.5 / 1_000_000
    cost_output_usd = output_tokens * 10.0 / 1_000_000
    total_cost_usd = cost_prompt_usd + cost_output_usd
    total_cost_pln = total_cost_usd * exchange_rate
    return total_cost_usd, total_cost_pln

# -------------------------  URUCHOMIENIE  -------------------------
def main():
    st.title("Nauka poprzez zabawne opowieści (z rejestracją i logowaniem)")

    # Upewniamy się, że istnieje kolekcja 'users'
    create_users_collection()

    # Inicjalizacja zmiennych sesyjnych
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    # Trzy zakładki: Logowanie/Rejestracja, Generowanie, Historia
    tab_auth, tab_generate, tab_saved = st.tabs(["Logowanie / Rejestracja", "Generuj", "Zapisane Opowieści i Żarty"])

    # -------------------------  Zakładka 1: Logowanie / Rejestracja  -------------------------
    with tab_auth:
        st.subheader("Zarejestruj się lub zaloguj")

        auth_mode = st.radio("Wybierz akcję:", ["Zarejestruj", "Zaloguj"])
        username = st.text_input("Nazwa użytkownika:", key="auth_username")
        password = st.text_input("Hasło:", type="password", key="auth_password")

        if st.button("Dalej", key="auth_button"):
            if not username or not password:
                st.error("Podaj nazwę użytkownika i hasło!")
            else:
                if auth_mode == "Zarejestruj":
                    success = register_user(username, password)
                    if success:
                        st.success("Rejestracja udana! Możesz się teraz zalogować.")
                    else:
                        st.warning("Użytkownik o tej nazwie już istnieje.")
                else:
                    # Logowanie
                    if login_user(username, password):
                        st.success("Zalogowano pomyślnie!")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        # Tworzymy kolekcję dla użytkownika (jeśli nie istnieje)
                        create_user_collection_if_not_exists(username)
                    else:
                        st.error("Błędna nazwa użytkownika lub hasło.")

    # -------------------------  Zakładka 2: Generowanie  -------------------------
    with tab_generate:
        st.header("Generowanie opowieści lub żartów")
        if not st.session_state["logged_in"]:
            st.warning("Musisz się zalogować, aby korzystać z tej zakładki.")
        else:
            st.info(f"Zalogowano jako: {st.session_state['username']}")

            problem_input = st.text_area("Wpisz problem lub cokolwiek, czego nie rozumiesz:")
            mode = st.radio("Wybierz tryb:", ("Ulubione Universum", "Zabawnie"))
            style_input = None
            if mode == "Ulubione Universum":
                style_input = st.text_input("Wpisz ulubionego pisarza, reżysera lub tytuł ulubionego filmu:")

            if st.button("Generuj", key="generate_button"):
                if not problem_input:
                    st.error("Pole problem jest puste!")
                else:
                    prompt = generate_prompt(problem_input, mode, style_input)
                    st.subheader("Wygenerowany prompt:")
                    st.code(prompt, language="python")

                    generated_text = get_gpt4_response(prompt)
                    if generated_text:
                        st.subheader("Wygenerowany tekst:")
                        st.write(generated_text)
                        
                        cost_usd, cost_pln = calculate_cost(prompt, generated_text)
                        st.markdown("---")
                        st.write(f"Szacowany koszt wywołania API GPT-4: {cost_usd*100:.2f} centów USD (~{cost_pln:.2f} zł)")

                        # Zapisujemy historię w kolekcji użytkownika
                        save_history(
                            account_name=st.session_state["username"],
                            record_type=mode,
                            prompt_text=prompt,
                            generated_text=generated_text,
                            cost_usd=cost_usd,
                            cost_pln=cost_pln
                        )

    # -------------------------  Zakładka 3: Przegląd zapisanych historii  -------------------------
    with tab_saved:
        st.header("Zapisane Opowieści i Żarty")
        if not st.session_state["logged_in"]:
            st.warning("Musisz się zalogować, aby przeglądać zapisaną historię.")
        else:
            st.info(f"Zalogowano jako: {st.session_state['username']}")
            st.write("Poniżej wyświetlamy zapisane rekordy dla zalogowanego użytkownika.")

            # Wczytujemy historię z kolekcji zalogowanego użytkownika
            user_collection = st.session_state["username"]
            client = get_qdrant_client()
            existing = client.get_collections().collections
            collection_names = [col.name for col in existing]

            if user_collection in collection_names:
                history = get_history(user_collection)
                if history:
                    for item in history:
                        payload = item.payload
                        st.write("---")
                        st.write(f"**Typ**: {payload.get('type')}")
                        st.write(f"**Prompt**: {payload.get('prompt')}")
                        st.write(f"**Tekst**: {payload.get('generated_text')}")
                        cost_usd = payload.get("cost_usd", 0)
                        cost_pln = payload.get("cost_pln", 0)
                        st.write(f"**Koszt**: {cost_usd*100:.2f} centów USD (~{cost_pln:.2f} zł)")
                else:
                    st.info("Brak zapisanych rekordów w tej kolekcji.")
            else:
                st.warning("Twoja kolekcja jeszcze nie istnieje lub nie została poprawnie utworzona.")


# -------------------------  URUCHOMIENIE  -------------------------
if __name__ == "__main__":
    main()
